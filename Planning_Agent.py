import tensorflow.compat.v1 as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from torch.autograd import Variable

logging.basicConfig(filename='Planning_Agent.log', level=logging.DEBUG, filemode='w')
from Agents import Agent

RANDOM_SEED = 5
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


class planning_network(nn.Module):
    def __init__(self):
        super(planning_network, self).__init__()
        self.fc1 = nn.Linear(2, 2)

    def forward(self, x):  # x:input actions
        # x = torch.from_numpy(x).to(torch.float32)
        out = self.fc1(x)
        out = torch.sigmoid(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.1)


class Planning_Agent(Agent):
    def __init__(self, env, underlying_agents, learning_rate=0.01,
                 gamma=0.95, max_reward_strength=None, cost_param=0, with_redistribution=False,
                 value_fn_variant='exact'):
        super().__init__(env, learning_rate, gamma)
        self.vp = None
        self.a_plan = None
        self.p_player = None
        self.p_players = None
        self.underlying_agents = underlying_agents
        self.log = []
        self.max_reward_strength = max_reward_strength
        n_players = len(underlying_agents)
        self.with_redistribution = with_redistribution
        self.value_fn_variant = value_fn_variant
        self.network = planning_network()
        self.optimizer = torch.optim.Adam(self.network.parameters(), learning_rate)
        self.cost_param = cost_param

    def learn(self, s, a_players):
        global loss
        s = s[np.newaxis, :]
        r_players = np.asarray(self.env.calculate_payoffs(a_players))
        a_players = np.asarray(a_players)
        # feed_dict = {self.s: s, self.a_players: a_players[np.newaxis, :],
        #              self.r_players: r_players[np.newaxis, :]}

        action_layer = self.network.forward(torch.from_numpy(a_players).to(torch.float32))
        self.vp = 2 * self.max_reward_strength * (action_layer - 0.5)

        if True:
            p_players_list = []
            for underlying_agent in self.underlying_agents:
                idx = underlying_agent.agent_idx
                p_players_list.append(underlying_agent.calc_action_probs(s)[0, -1].detach().numpy())
            p_players_arr = np.reshape(np.asarray(p_players_list), [1, -1])
            # feed_dict[self.p_players] = p_players_arr
            # feed_dict[self.a_plan] = self.calc_conditional_planning_actions(s)

            self.p_player = p_players_arr
            self.a_plan = self.calc_conditional_planning_actions(s)

            cost_list = []
            for underlying_agent in self.underlying_agents:
                # policy gradient theorem
                idx = underlying_agent.agent_idx

                if True:
                    # 猜测是对P(C)这个sigmoid对theta_idx求导
                    self.g_p = p_players_arr[0, idx] * (1 - p_players_arr[0, idx])
                    # 猜测是算对手的P(C)
                    self.p_opp = p_players_arr[0, 1 - idx]

                    # 可能是链式法则，先让vp对P(C)求导，再让P(C)对theta_idx求导
                    a_players_var = Variable(torch.from_numpy(a_players).to(torch.float32), requires_grad=True)
                    action_layer_var = self.network.forward(a_players_var)
                    self.vp_var = 2 * self.max_reward_strength * (action_layer_var - 0.5)
                    test = torch.autograd.grad(self.vp_var[idx], a_players_var)[0][idx]
                    self.g_Vp = self.g_p * test


                    # 这里的V是r1+r2，先对P(C)求导，再让P(C)对theta_idx求导
                    self.g_V = self.g_p * (self.p_opp * (2 * self.env.R - self.env.T - self.env.S)
                                           + (1 - self.p_opp) * (self.env.T + self.env.S - 2 * self.env.P))

                # cost_list.append(- underlying_agent.learning_rate * tf.tensordot(self.g_Vp,self.g_V,1))
                cost_list.append(- underlying_agent.learning_rate * self.g_Vp * self.g_V)

            if self.with_redistribution:
                # 分配的rewards总和为0
                extra_loss = self.cost_param * torch.norm(self.vp - torch.mean(self.vp))
            else:
                extra_loss = self.cost_param * torch.norm(self.vp)
            # 这里的loss是负的需要最大化的objective，即最小化负值
            loss = torch.sum(torch.stack(cost_list)) + extra_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        action = action_layer
        g_Vp = self.g_Vp
        g_V = self.g_V

        logging.info('Learning step')
        logging.info('Planning_action: ' + str(action))
        logging.info('Gradient of V_p: ' + str(g_Vp))
        logging.info('Gradient of V: ' + str(g_V))
        logging.info('Loss: ' + str(loss))

    def get_log(self):
        return self.log

    def choose_action(self, s, a_players):
        # simple action 直接根据概率选的actions
        logging.info('Player actions: ' + str(a_players))
        s = s[np.newaxis, :]
        a_players = np.asarray(a_players)
        # a_plan = self.sess.run(self.action_layer, {self.s: s, self.a_players: a_players[np.newaxis, :]})[0, :]
        a_plan = self.network.forward(torch.from_numpy(a_players).to(torch.float32))
        if self.max_reward_strength is not None:
            a_plan = 2 * self.max_reward_strength * (a_plan - 0.5)
        logging.info('Planning action: ' + str(a_plan))
        # 四种情况下agent_1得到的planning分配Vp1，画图用
        self.log.append(self.calc_conditional_planning_actions(s))
        # a_plan指的是Planning基于当前action给出的Vp，给两个agents
        return a_plan

    def calc_conditional_planning_actions(self, s):
        # Planning actions in each of the 4 cases: DD, CD, DC, CC
        # a_plan_DD = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([0, 0])[np.newaxis, :]})
        # a_plan_CD = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([1, 0])[np.newaxis, :]})
        # a_plan_DC = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([0, 1])[np.newaxis, :]})
        # a_plan_CC = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([1, 1])[np.newaxis, :]})
        #
        # l_temp = [a_plan_DD, a_plan_CD, a_plan_DC, a_plan_CC]
        #
        # if self.max_reward_strength is not None:
        #     l = [2 * self.max_reward_strength * (a_plan_X[0, 0] - 0.5) for a_plan_X in l_temp]
        # else:
        #     l = [a_plan_X[0, 0] for a_plan_X in l_temp]
        # if self.with_redistribution:
        #     if self.max_reward_strength is not None:
        #         l2 = [2 * self.max_reward_strength * (a_plan_X[0, 1] - 0.5) for a_plan_X in l_temp]
        #     else:
        #         l2 = [a_plan_X[0, 1] for a_plan_X in l_temp]
        #     l = [0.5 * (elt[0] - elt[1]) for elt in zip(l, l2)]
        test = [[-0.13079613 -0.37225109], [-0.25362432 -0.4930197 ]]
        return test
