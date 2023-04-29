import numpy as np
import tensorflow.compat.v1 as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(filename='Agents.log', level=logging.DEBUG)

RANDOM_SEED = 8
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

from enum import Enum, auto

tf.disable_v2_behavior()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Critic_Variant(Enum):
    INDEPENDENT = auto()
    CENTRALIZED = auto()
    CENTRALIZED_APPROX = auto()


class Agent(object):
    def __init__(self, env, learning_rate=0.001, gamma=0.95, agent_idx=0):
        self.sess = tf.Session()
        self.env = env
        self.n_actions = env.n_actions
        self.n_features = env.n_features
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.agent_idx = agent_idx
        self.log = []  # logs action probabilities

    def choose_action(self, s):
        action_probs = self.calc_action_probs(s).detach().numpy()
        action = np.random.choice(range(action_probs.shape[0]),
                                  p=action_probs.ravel())  # select action w.r.t the actions prob
        self.log.append(action_probs[1]) # 记录该simple_agent此时选择合作的概率
        return action

    def learn_at_episode_end(self):
        pass

    def close(self):
        self.sess.close()
        tf.reset_default_graph()

    def reset(self):
        self.sess.run(tf.global_variables_initializer())


class Actor_Critic_Agent(Agent):
    def __init__(self, env, learning_rate=0.001, n_units_actor=20,
                 n_units_critic=20, gamma=0.95, agent_idx=0,
                 critic_variant=Critic_Variant.INDEPENDENT, *args):
        super().__init__(env, learning_rate, gamma, agent_idx)
        self.actor = Actor(env, n_units_actor, learning_rate, agent_idx)
        self.critic = Critic(env, n_units_critic, learning_rate, gamma, agent_idx,
                             critic_variant)
        self.sess.run(tf.global_variables_initializer())

    def learn(self, s, a, r, s_, done=False, *args):
        if done:
            pass
        else:
            td = self.critic.learn(self.sess, s, r, s_, *args)
            self.actor.learn(self.sess, s, a, td)

    def __str__(self):
        return 'Actor_Critic_Agent_' + str(self.agent_idx)

    def calc_action_probs(self, s):
        return self.actor.calc_action_probs(self.sess, s)

    def pass_agent_list(self, agent_list):
        self.critic.pass_agent_list(agent_list)

    def get_action_prob_variable(self):
        return self.actor.actions_prob

    def get_state_variable(self):
        return self.actor.s

    def get_policy_parameters(self):
        return [self.actor.w_l1, self.actor.b_l1, self.actor.w_pi1, self.actor.b_pi1]


class actor_network(nn.Module):
    def __init__(self, env):
        super(actor_network, self).__init__()
        self.fc1 = nn.Linear(env.n_features, 20)
        self.fc2 = nn.Linear(20, env.n_actions)

    def forward(self, x):  # x:input state
        x = torch.from_numpy(x[0]).to(torch.float32)
        out = F.relu(self.fc1(x))
        fc2 = F.softmax(self.fc2(out))
        acts_prob = fc2
        return acts_prob

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.1)

class Actor(object):
    def __init__(self, env, n_units=20, learning_rate=0.001, agent_idx=0):
        self.actions_prob = None
        self.td_error = None
        self.s = None
        self.a = None
        self.network = actor_network(env)

        self.optimizer = torch.optim.Adam(self.network.parameters(), learning_rate)


    def learn(self,s, a, td):
        """ Actor进行学习
        :param s: 状态
        :param a: 动作
        :param td: 来自critic，指导Actor对不对
        :return:
        """
        self.td_error = torch.tensor(td).detach()
        log_prob = torch.log(self.actions_prob[a])
        loss = torch.mean(-log_prob*self.td_error)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss



class critic_network(nn.Module):
    def __init__(self, env):
        super(critic_network, self).__init__()
        self.fc1 = nn.Linear(env.n_features+env.n_actions * env.n_players, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        v_ = self.fc2(out)
        return v_

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.1)

class Critic(object):
    def __init__(self, env, n_units, learning_rate, gamma, agent_idx,
                 critic_variant=Critic_Variant.INDEPENDENT):
        self.critic_variant = critic_variant
        self.env = env
        self.network = critic_network(env)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.v = None
        self.v_= None
        # self.td_error = None
        self.loss = None

    def pass_agent_list(self, agent_list):
        self.agent_list = agent_list

    def learn(self, sess, s, r, s_, *args):
        # s, s_ = s.astype(np.float32), s_.astype(np.float32)
        act_probs = torch.hstack([agent.calc_action_probs(s) for idx, agent in enumerate(self.agent_list)])
        act_probs_ = torch.hstack([agent.calc_action_probs(s_) for idx, agent in enumerate(self.agent_list)])
        nn_inputs = torch.hstack([torch.from_numpy(s), act_probs])
        nn_inputs_ = torch.hstack([torch.from_numpy(s_), act_probs_])
        r = torch.tensor(r, requires_grad=False)

        self.v = self.network.forward(nn_inputs.to(torch.float32))
        self.v_= self.network.forward(nn_inputs_.to(torch.float32))
        td_error = r + self.gamma * self.v_ - self.v
        td_error = torch.square(td_error)

        self.optimizer.zero_grad()  # 清空梯度
        td_error.backward(retain_graph=True)
        self.optimizer.step()

        return td_error


class Simple_Agent(Agent):  # plays games with 2 actions, using a single parameter
    def __init__(self, env, learning_rate=0.001, n_units_critic=20, gamma=0.95, agent_idx=0,
                 critic_variant=Critic_Variant.INDEPENDENT):
        super().__init__(env, learning_rate, gamma, agent_idx)
        # self.s = tf.placeholder(tf.float32, [1, env.n_features], "state")  # dummy variable
        # self.a = tf.placeholder(tf.int32, None, "act")
        # self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.s = None
        self.a = None
        self.td_error = None


        self.critic = Critic(env, n_units_critic, learning_rate, gamma, agent_idx, critic_variant)
        self.actor = Actor(env, n_units=20, learning_rate=0.001, agent_idx=agent_idx)

        # self.sess.run(tf.global_variables_initializer())

    def learn(self, s, a, r, s_, done=False, *args):
        if done:
            pass
        else:
            td = self.critic.learn(self.sess, s, r, s_, *args)
            self.actor.learn(s, a, td)
            # feed_dict = {self.a: a, self.td_error: td}
            # _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)

    def __str__(self):
        return 'Simple_Agent_' + str(self.agent_idx)

    def calc_action_probs(self, s):
        s = s[np.newaxis, :]
        probs = self.actor.network.forward(s)
        self.actor.actions_prob = probs
        # probs = self.sess.run(self.actions_prob)
        return probs

    def pass_agent_list(self, agent_list):
        self.critic.pass_agent_list(agent_list)

    # def get_state_variable(self):
    #     return self.s

    # def calc_g_log_pi(self, s, a):
    #     return self.sess.run(self.g_log_pi, feed_dict={self.s: s, self.a: a})
