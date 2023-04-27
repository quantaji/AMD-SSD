from ray.rllib.algorithms.callbacks import DefaultCallbacks

# typing:
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from .amd_torch_policy import AMDAgentPolicy, AMDAgentTorchPolicy
from .amd import PreLearningProcessing


class AMDAgentWarenessCallback(DefaultCallbacks):
    """
    This callback:
        1. generate the single agents' awareness. (sum_t V_t g_log_p_t)^T(sum_t g_log_p_t), of shape (T, ). 
        2. it generate the SampleBatch needed for the central planner.
        3. forward these to central planner and get the planner's rewards r_planner, and update each single agent's sample batch
    """

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs) -> None:

        # ! STEP 1: get each agent's awareness at each timestep
        if isinstance(samples, MultiAgentBatch):
            for pid, batch in samples.policy_batches.items():
                batch.decompress_if_needed()
                policy: AMDAgentPolicy = worker.policy_map[pid]

                print(
                    '-----*****-----',
                    pid,
                    batch,
                    batch[SampleBatch.T],
                    # batch[SampleBatch.EPS_ID],
                    # batch[SampleBatch.UNROLL_ID],
                    # batch[PreLearningProcessing.R_PLANNER]
                )
                policy.config['policy_param'] = 'neural'
        else:
            raise ValueError("This should be a multi-agent sample batch, check your algorithm configuration and environment!")
        # # print(type(samples), samples, worker.policy_dict, worker.policy_map, worker.policy_config)

        # MultiAgentBatch({'wolf_1': SampleBatch(128: ['obs', 'actions', 'rewards', 'terminateds', 'truncateds', 'infos', 'eps_id', 'unroll_id', 'agent_index', 't', 'advantages', 'value_targets']),

        # TODO ! STEP 2: Generate obs for central planner, and get each agent's awareness as a dict.

        # raise NotImplementedError

        # TODO ! STEP 3: update each single agent's sample batch
