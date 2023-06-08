import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import ray
import torch
import torch.nn as nn
from gymnasium import spaces
from numpy import ndarray
from ray.rllib.algorithms.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import (Postprocessing, compute_gae_for_sample_batch)
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch import torch_modelv2
from ray.rllib.models.torch.torch_action_dist import (TorchCategorical, TorchDistributionWrapper)
from ray.rllib.policy import Policy
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (EntropyCoeffSchedule, KLCoeffMixin, LearningRateSchedule, ValueNetworkMixin)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (apply_grad_clipping, explained_variance, sequence_mask, warn_if_infinite_kl_divergence)
from ray.rllib.utils.typing import AgentID, TensorStructType, TensorType
from torch.autograd import grad
from torch.func import functional_call, jacrev

from ..amd.amd_torch_policy import AMDAgentTorchPolicy, AMDGeneralPolicy
from ..amd.constants import PreLearningProcessing
from ..amd.utils import (action_to_reward, chunk_batch_to_batch, cumsum_factor_across_eps)

torch, nn = try_import_torch()


class PlannerRewardCoeffSchedule:
    """A sigmoid factor with two parameter denoting the halp time step and time-scale"""

    @DeveloperAPI
    def __init__(self, half_step: int, step_scale: int):
        self.half_step = half_step
        self.step_scale = step_scale

        self.cur_planner_reward_coeff = self.planner_reward_coeff(0)

    def planner_reward_coeff(self, step):
        return 1 / (1 + np.exp(-(step - self.half_step) / self.step_scale))

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(PlannerRewardCoeffSchedule, self).on_global_var_update(global_vars)
        self.cur_planner_reward_coeff = self.planner_reward_coeff(global_vars["timestep"])


class AMDPPOAgentTorchPolicy(
        AMDGeneralPolicy,
        ValueNetworkMixin,
        LearningRateSchedule,
        EntropyCoeffSchedule,
        PlannerRewardCoeffSchedule,
        KLCoeffMixin,
        TorchPolicyV2,
):

    model: nn.Module
    agent_info_space: spaces.Space

    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.algorithms.ppo.ppo.PPOConfig().to_dict(), **config)
        # TODO: Move into Policy API, if needed at all here. Why not move this into
        #  `PPOConfig`?.
        validate_config(config)

        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        ValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(self, config["entropy_coeff"], config["entropy_coeff_schedule"])

        PlannerRewardCoeffSchedule.__init__(
            self,
            config["pfactor_half_step"],
            config["pfactor_step_scale"],
        )

        KLCoeffMixin.__init__(self, config)

        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        return AMDAgentTorchPolicy.postprocess_trajectory(self, sample_batch, other_agent_batches, episode)

    def compute_central_planner_actions(self, sample_batch: SampleBatch) -> ndarray:
        return AMDAgentTorchPolicy.compute_central_planner_actions(self, sample_batch)

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        if self.is_central_planner:
            return convert_to_numpy({
                "planner_policy_loss": torch.mean(torch.stack(self.get_tower_stats("planner_policy_loss"))),
                "planner_reward_cost": torch.mean(torch.stack(self.get_tower_stats("planner_reward_cost"))),
                "planner_reward_std": torch.mean(torch.stack(self.get_tower_stats("planner_reward_std"))),
                "cur_lr": self.cur_lr,
            })
        else:
            # get a stats on the cooperativeness of current batch
            coop_stats_fn = self.config['agent_cooperativeness_stats_fn']
            cooperativeness = 0.0
            if coop_stats_fn:
                cooperativeness = coop_stats_fn(train_batch)

            return convert_to_numpy({
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(torch.stack(self.get_tower_stats("total_loss"))),
                "policy_loss": torch.mean(torch.stack(self.get_tower_stats("mean_policy_loss"))),
                "vf_loss": torch.mean(torch.stack(self.get_tower_stats("mean_vf_loss"))),
                "vf_explained_var": torch.mean(torch.stack(self.get_tower_stats("vf_explained_var"))),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(torch.stack(self.get_tower_stats("mean_entropy"))),
                "entropy_coeff": self.entropy_coeff,
                "amd_loss": torch.mean(torch.stack(self.get_tower_stats("amd_loss"))),
                "planner_reward_coeff": self.cur_planner_reward_coeff,
                "cooperativeness": cooperativeness,
            })

    def loss_central_planner(self, model: ModelV2, dist_class: ActionDistribution, train_batch: SampleBatch) -> TensorType | List[TensorType]:
        # Pass the training data through our model to get distribution parameters.
        dist_inputs, _ = model(train_batch)

        if self.is_recurrent():
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = train_batch[SampleBatch.REWARDS].shape[0] // B
            mask_orig = sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
            valid_mask = torch.reshape(mask_orig, [-1])
        else:
            valid_mask = torch.ones_like(train_batch[SampleBatch.REWARDS], dtype=torch.bool)

        policy_loss = torch.zeros_like(dist_inputs, requires_grad=True).mean()
        reward_cost = torch.zeros_like(policy_loss, requires_grad=True)
        reward_std = torch.zeros_like(policy_loss, requires_grad=True)

        # only compute loss when reward_max is not zero, for saving computation
        if self.config['planner_reward_max'] > 0.0:

            # Create an action distribution object.
            action_dist = dist_class(dist_inputs[valid_mask], model)

            # Compute the actual actions
            actions = action_dist.deterministic_sample()

            # get reward from actions
            r_planner = action_to_reward(
                actions=actions,
                appearance=self.appearance,
                reward_max=self.config['planner_reward_max'],
                zero_sum=self.config['force_zero_sum'],
                availability=train_batch[PreLearningProcessing.AVAILABILITY][valid_mask],
            )
            use_cum_reward = False
            if self.config['use_cum_reward']:
                # get the matrix for computing accumulated reward
                # fix: somehow on clusters, discoutned factor matrix is not T x T
                cp_eps_id = train_batch[SampleBatch.EPS_ID][valid_mask]

                # we project it onto per episode scale
                factor_matrix = cumsum_factor_across_eps(cp_eps_id).to(r_planner)
                r_planner_cum = factor_matrix @ r_planner
                awareness_cum = factor_matrix @ train_batch[PreLearningProcessing.AWARENESS][valid_mask]
            else:
                r_planner_cum = r_planner
                awareness_cum = train_batch[PreLearningProcessing.AWARENESS][valid_mask]

            policy_loss = -torch.mean(awareness_cum * r_planner_cum) * self.config['agent_pseudo_lr']
            reward_cost = ((r_planner**2).mean(-1)**0.5).mean()
            reward_std = r_planner.std()

        model.tower_stats["planner_policy_loss"] = policy_loss
        model.tower_stats["planner_reward_cost"] = reward_cost
        model.tower_stats["planner_reward_std"] = reward_std

        total_loss = policy_loss + self.config['planner_reward_cost'] * reward_cost

        return total_loss

    def loss_agent(self, model: ModelV2, dist_class: ActionDistribution, train_batch: SampleBatch) -> TensorType | List[TensorType]:

        dist_inputs, state = model(train_batch)
        curr_action_dist = dist_class(dist_inputs, model)

        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = dist_inputs.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)

        logp = curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])

        # # amd loss computed here
        amd_loss = -logp * train_batch[PreLearningProcessing.R_PLANNER_CUM]

        logp_ratio = torch.exp(logp - train_batch[SampleBatch.ACTION_LOGP])

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            # TODO smorad: should we do anything besides warn? Could discard KL term
            # for this update
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES] * torch.clamp(logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]),
        )

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        total_loss = reduce_mean_valid(-surrogate_loss + self.config["vf_loss_coeff"] * vf_loss_clipped - self.entropy_coeff * curr_entropy)

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        if self.config["planner_reward_max"] > 0.0:
            total_loss += self.cur_planner_reward_coeff * reduce_mean_valid(amd_loss)

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(train_batch[Postprocessing.VALUE_TARGETS], value_fn_out)
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss
        model.tower_stats["amd_loss"] = reduce_mean_valid(amd_loss)

        return total_loss

    @override(TorchPolicyV2)
    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    def compute_awareness(self, sample_batch: SampleBatch) -> SampleBatch:
        """This funciton compute the 'awareness' of an agent.

            The awareness is a intermediate value used in total value function approximation. In the policy gradient approximation, 
                V_j = Prod_{i,t} pi_i(t) x R_j.
            and the gradient is 
                nabla_i V_j = (sum_t nabla_i log pi_i(t)) x R_j.
            and policy gradient theorem, we can replace R_j with advantage function
                nabla_i V_j = sum_t ( (nabla_i log pi_i(t)) x delta_{j, t} )
            Then the total value function's gradient is
                nabla_i V = sum_t ( (nabla_i log pi_i(t)) x (sum_j delta_{j, t}) )
            The AWARENESS is defined as
                a_{t,i} = (nabla_i V)^T x (nabla_i log pi_i(t))
                        = sum_t' ( (nabla_i log pi_i(t')) x (sum_j delta_{j, t'}) )^T x (nabla_i log pi_i(t))
            The awareness is used for latter update of central planner
                Delta theta_p = sum_{i, t} a_{t, i} x nabla_p R^p_{t, i}
            where R^p_{t,i} is the accumlated total reward.
        """
        # planner_reward_max means no planner's reward, so we skip this for faster computation
        if self.is_central_planner or self.config['planner_reward_max'] == 0.0:
            return sample_batch

        awareness = None

        if not self.config['awareness_batch_size']:
            batch_slices = [sample_batch]
        else:
            slice_chunk = chunk_batch_to_batch(
                eps_id=sample_batch[SampleBatch.EPS_ID],
                batch_size=self.config['awareness_batch_size'],
            )
            batch_slices = [sample_batch.slice(slice_chunk[i], slice_chunk[i + 1]) for i in range(len(slice_chunk) - 1)]

        def get_copied_batch_and_mask(train_batch: SampleBatch):
            # get ready for calculating awareness
            copied_batch = train_batch.copy(shallow=False)
            if not copied_batch.zero_padded:
                pad_batch_to_sequences_of_same_size(
                    batch=copied_batch,
                    max_seq_len=self.max_seq_len,
                    shuffle=False,
                    batch_divisibility_req=self.batch_divisibility_req,
                    view_requirements=self.view_requirements,
                )
            copied_batch = self._lazy_tensor_dict(copied_batch)  # this also convert batch to device

            if self.is_recurrent():
                B = len(copied_batch[SampleBatch.SEQ_LENS])
                max_seq_len = copied_batch[SampleBatch.REWARDS].shape[0] // B
                mask_orig = sequence_mask(copied_batch[SampleBatch.SEQ_LENS], max_seq_len)
                valid_mask = torch.reshape(mask_orig, [-1])
            else:
                valid_mask = torch.ones_like(copied_batch[SampleBatch.REWARDS], dtype=torch.bool)

            return copied_batch, valid_mask

        if self.config['param_assumption'] == 'neural':
            """
            Agents do policy gradient with assumtion of neural network parameterization
            """
            awareness = []

            # I have tested, grad takes less VRAM
            if self.config['neural_awareness_method'] == 'jacobian':
                """
                Use torch.func.jacrev to calculate awarenss
                """
                dict_params = {k: v.detach() for k, v in self.model.named_parameters()}
                policy_gradient = {k: np.zeros(v.shape) for k, v in dict_params.items()}

                # first loop for calculation of gradient
                for batch in batch_slices:
                    copied_batch, valid_mask = get_copied_batch_and_mask(batch)

                    def func_cal_policy_loss(params):
                        dist_inputs, _ = functional_call(self.model, params, copied_batch)
                        action_dist = self.dist_class(dist_inputs, self.model)
                        log_probs = action_dist.logp(copied_batch[SampleBatch.ACTIONS]).reshape(-1)  # shape (Batch, )
                        return torch.sum(log_probs[valid_mask] * copied_batch[PreLearningProcessing.TOTAL_ADVANTAGES][valid_mask])

                    batch_pg = jacrev(func_cal_policy_loss)(dict_params)

                    with torch.no_grad():
                        for k, pg_params in batch_pg.items():
                            policy_gradient[k] += pg_params.detach().cpu().numpy()

                # second loop for calculating awarenss
                for batch in batch_slices:
                    batch_awareness = np.zeros_like(batch[SampleBatch.REWARDS], dtype=np.float).reshape(-1)
                    batch_size = batch_awareness.shape[0]

                    copied_batch, valid_mask = get_copied_batch_and_mask(batch)

                    def func_cal_log_prob(params):
                        dist_inputs, _ = functional_call(self.model, params, copied_batch)
                        action_dist = self.dist_class(dist_inputs, self.model)
                        log_probs = action_dist.logp(copied_batch[SampleBatch.ACTIONS]).reshape(-1)  # shape (Batch, )
                        return log_probs[valid_mask]

                    batch_log_prob = jacrev(func_cal_log_prob)(dict_params)
                    with torch.no_grad():
                        for k, logp_params in batch_log_prob.items():
                            # log_params is of shape (batch, (shape_params))
                            batch_awareness += (logp_params.detach().cpu().numpy() * policy_gradient[k]).reshape(batch_size, -1).sum(-1)

                    awareness.append(batch_awareness)

                awareness = np.hstack(awareness)

            elif self.config['neural_awareness_method'] == 'grad':
                """
                    Use torch.autograd.grad to calculate, needs more memory because have to create graph
                """
                policy_gradient = [torch.zeros_like(p) for p in self.model.parameters()]

                self.model.train()
                for batch in batch_slices:
                    copied_batch, valid_mask = get_copied_batch_and_mask(batch)
                    dist_inputs, _ = self.model(copied_batch)
                    action_dist = self.dist_class(dist_inputs, self.model)
                    log_probs = action_dist.logp(copied_batch[SampleBatch.ACTIONS]).reshape(-1)  # shape (Batch, )
                    policy_loss = torch.sum(log_probs[valid_mask] * copied_batch[PreLearningProcessing.TOTAL_ADVANTAGES][valid_mask])

                    batch_pg = grad(policy_loss, self.model.parameters(), allow_unused=True)

                    for i, pg_params in enumerate(policy_gradient):
                        if batch_pg[i] is not None:
                            pg_params += batch_pg[i]

                for batch in batch_slices:
                    batch_awareness = np.zeros_like(batch[SampleBatch.REWARDS], dtype=np.float).reshape(-1)
                    batch_size = batch_awareness.shape[0]

                    copied_batch, valid_mask = get_copied_batch_and_mask(batch)
                    with torch.backends.cudnn.flags(enabled=False):
                        # because cudnn does not have second derivative of rnns
                        dist_inputs, _ = self.model(copied_batch)
                    action_dist = self.dist_class(dist_inputs, self.model)
                    log_probs = action_dist.logp(copied_batch[SampleBatch.ACTIONS]).reshape(-1)[valid_mask]  # shape (Batch, )
                    ones = torch.ones_like(log_probs, requires_grad=True)

                    logp_sum = grad(log_probs, self.model.parameters(), ones, create_graph=True, allow_unused=True)

                    for i, pg_params in enumerate(policy_gradient):
                        if logp_sum[i] is not None:
                            batch_awareness += grad(
                                (pg_params * logp_sum[i]).sum(),
                                ones,
                                retain_graph=True,
                            )[0].detach().cpu().numpy()

                    awareness.append(batch_awareness)

                awareness = np.hstack(awareness)

            else:
                raise NotImplementedError

        elif self.config['param_assumption'] == 'softmax':
            """
            Agents do policy gradient with assumtion of softmax parameterizization. In this case:
                p(a, s) = softmax(theta(a', s)) (a)
                paital{logp(a, s)}{theta(a', s')} = 1{s=s', a=a'} - pi(a', s)1{s=s'}
            NOTE: 'softmax' option natually assumes a multinomial distribution, so the model's output is assumed to be the logits.
            NOTE: Determining state is equal or not is difficult. Therefore, I use hash function over observation.
            Derivation:
                awareness(tau) := (sum_t nabla_theta logpi(a_t, s_t) * delta_t) ^T nabla_theta logpi(a_tau, s_tau)
                = sum_{t, a', s'} delta_t x 1{s_t = s'} x (1{a_t = a'} - pi(a', s_t)) x 1{s_tau = s'} x (1{a_tau = a'} - pi(a', s_tau))
            We define j_t = 1{a_t = a'} - pi(a', s_t) in R^{|A|} as a vector of dimension of actiona space, then
                awareness(tau) = sum_{t, s'} delta_t x 1{s_t = s'} x 1{s_tau = s'} x (j_t ^T j_tau)
                =  sum_{t} delta_t x 1{s_t = s_tau} x (j_t ^T j_tau)
                = M x delta
            M_{i,j} := 1{s_i = s_j} x (j_i ^T j_j) as a TxT matrix, and delta = [delta_t] is a Tx1 matrix
            """
            assert self.dist_class == TorchCategorical, "Softmax parameterization assumtion assumse tabular case, where action is discrete and therefore, action distribution class is catagorical!"

            jac_logp_s = []

            with torch.no_grad():
                for batch in batch_slices:
                    copied_batch, valid_mask = get_copied_batch_and_mask(batch)

                    logits, _ = self.model(copied_batch)  # shape is (Batch, num_actions)
                    num_actions = logits.shape[1]

                    batch_jac_logp_s = torch.nn.functional.one_hot(copied_batch[SampleBatch.ACTIONS][valid_mask].long(), num_classes=num_actions) - torch.softmax(logits[valid_mask], dim=-1)  # (Batch, num_actions)
                    batch_jac_logp_s = batch_jac_logp_s.detach().cpu().numpy()

                    jac_logp_s.append(batch_jac_logp_s)

            jac_logp_s = np.concatenate(jac_logp_s)

            obs = sample_batch[SampleBatch.OBS]
            obs_hash = np.apply_along_axis(
                lambda arr: hash(arr.data.tobytes()),
                1,
                obs.reshape(obs.shape[0], -1),
            )  # this funciton computes the observation's hash

            # big matrix, we should work on cpu
            total_adv = sample_batch[PreLearningProcessing.TOTAL_ADVANTAGES].reshape(-1, 1)

            batch_size = obs_hash.shape[0]

            if not self.config['awareness_batch_size']:
                batch_indices = [np.arange(batch_size)]
            else:
                batch_indices = np.split(np.arange(batch_size), np.arange((batch_size - 1) // self.config['awareness_batch_size'])[1:] * self.config['awareness_batch_size'])

            awareness = []
            for batch_index in batch_indices:
                batch_obs_hash = obs_hash[batch_index]
                batch_jac_logp_s = jac_logp_s[batch_index]
                temp = (batch_obs_hash.reshape(-1, 1) == obs_hash.reshape(1, -1)) @ (total_adv * jac_logp_s)
                batch_awareness = (temp * batch_jac_logp_s).sum(-1)
                awareness.append(batch_awareness)

            awareness = np.concatenate(awareness)

        else:
            raise ValueError("The current policy parameterization assumption {} is not supported!!!".format(self.config['policy_param']))

        sample_batch[PreLearningProcessing.AWARENESS] = awareness

        return sample_batch
