import matplotlib.pyplot as plt
import numpy as np
import math
import os
import logging

logging.basicConfig(filename='main.log', level=logging.DEBUG, filemode='w')
from Environments import Matrix_Game
from Agents import Actor_Critic_Agent, Critic_Variant, Simple_Agent
from Planning_Agent import Planning_Agent

N_EPISODES = 4000
N_PLAYERS = 2
N_UNITS = 10  # number of nodes in the intermediate layer of the NN
MAX_REWARD_STRENGTH = 3


def run_game(N_EPISODES, players, action_flip_prob, planning_agent=None, with_redistribution=True,
             n_planning_eps=math.inf):
    env.reset_ep_ctr()
    avg_planning_rewards_per_round = []
    for episode in range(N_EPISODES):
        # initial observation
        s = env.reset()
        flag = isinstance(s, list)

        cum_planning_rs = [0] * len(players)
        while True:
            # choose action based on s
            if flag:
                actions = [player.choose_action(s[idx]) for idx, player in enumerate(players)]
            else:
                # 直接根据概率来选simple agent的actions
                actions = [player.choose_action(s) for player in players]

            # take action and get next s and reward
            # 感觉一步的matrix game里面state更新没什么用，另外这里的rewards是没有modified的matrix game的rewards
            s_, rewards, done = env.step(actions)

            perturbed_actions = [(1 - a if np.random.binomial(1, action_flip_prob) else a) for a in actions]
            # perturbed_actions = [0,0]

            if planning_agent is not None and episode < n_planning_eps:
                # planning_agent 分配的rewards
                planning_rs = planning_agent.choose_action(s, perturbed_actions)
                if with_redistribution:
                    sum_planning_r = sum(planning_rs)
                    mean_planning_r = sum_planning_r / N_PLAYERS
                    planning_rs = [r - mean_planning_r for r in planning_rs]
                # modify一下每一个agent的真实rewards
                rewards = [sum(r) for r in zip(rewards, planning_rs)]
                cum_planning_rs = [sum(r) for r in zip(cum_planning_rs, planning_rs)]
                # Training planning agent
                # learn一下planning agent的网络，要考虑extra loss
                planning_agent.learn(s, perturbed_actions)
            logging.info('Actions:' + str(actions)) # [a1, a2]
            logging.info('State after:' + str(s_))
            logging.info('Rewards: ' + str(rewards)) # [Vtot1, Vtot2]
            logging.info('Done:' + str(done))

            for idx, player in enumerate(players):
                if flag:
                    player.learn(s[idx], actions[idx], rewards[idx], s_[idx], s, s_)
                else:
                    # 对于一个agent，把修改后的reward和他的行为扔进去，用critic来评估td_error
                    player.learn(s, actions[idx], rewards[idx], s_)
            # swap s
            s = s_

            # break while loop when done
            if done:
                for player in players:
                    player.learn_at_episode_end()
                break
        # 这里的matrix game只有一个step的操作，所以其实就是一轮的planning agent分配的rewards
        avg_planning_rewards_per_round.append([r.detach().numpy() / env.step_ctr for r in cum_planning_rs])

        # status updates
        if (episode + 1) % 100 == 0:
            print('Episode {} finished.'.format(episode + 1))
    return env.get_avg_rewards_per_round(), np.asarray(avg_planning_rewards_per_round)


def plot_results(data, legend, path, title, ylabel='Reward', exp_factor=1):
    plt.figure()
    for agent_idx in range(data.shape[1]):
        avg = data[0, agent_idx]
        avg_list = []
        for r in data[:, agent_idx]:
            avg = exp_factor * r + (1 - exp_factor) * avg
            avg_list.append(avg)
        first_idx = int(1 / exp_factor)
        plt.plot(range(first_idx, len(avg_list)), avg_list[first_idx:])
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.title(title)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + '/' + title)
    # plt.show()


def create_population(env, n_agents, use_simple_agents=False):
    critic_variant = Critic_Variant.CENTRALIZED
    if use_simple_agents:
        l = [Simple_Agent(env,
                          learning_rate=0.01,
                          gamma=0.9,
                          agent_idx=i,
                          critic_variant=critic_variant) for i in range(n_agents)]
    else:
        l = [Actor_Critic_Agent(env,
                                learning_rate=0.01,
                                gamma=0.9,
                                n_units_actor=N_UNITS,
                                agent_idx=i,
                                critic_variant=critic_variant) for i in range(n_agents)]
    # Pass list of agents for centralized critic
    if critic_variant is Critic_Variant.CENTRALIZED:
        for agent in l:
            agent.pass_agent_list(l)
    return l


def run_game_and_plot_results(env, agents,
                              with_redistribution=False, max_reward_strength=None, cost_param=0,
                              n_planning_eps=math.inf, value_fn_variant='exact', action_flip_prob=0):
    planning_agent = Planning_Agent(env, agents, max_reward_strength=max_reward_strength,
                                    cost_param=cost_param, with_redistribution=with_redistribution,
                                    value_fn_variant=value_fn_variant)
    avg_rewards_per_round, avg_planning_rewards_per_round = run_game(N_EPISODES, agents, action_flip_prob,
                                                                     planning_agent=planning_agent,
                                                                     with_redistribution=with_redistribution,
                                                                     n_planning_eps=n_planning_eps)
    path = './Results/' + env.__str__() + '/with' + ('' if with_redistribution else 'out') + '_redistribution'
    path += '/' + 'max_reward_strength_' + (str(max_reward_strength) if max_reward_strength is not None else 'inf')
    path += '/' + 'cost_parameter_' + str(cost_param)
    path += '/' + value_fn_variant + '_value_function'
    if n_planning_eps < math.inf:
        path += '/' + 'turning_off'
    if action_flip_prob > 0:
        path += '/' + 'action_flip_prob' + str(action_flip_prob)

    plot_results(avg_rewards_per_round, [str(agent) for agent in agents], path, 'average_rewards', exp_factor=0.05)
    plot_results(avg_planning_rewards_per_round, [str(agent) for agent in agents], path, 'planning_rewards',
                 exp_factor=0.05)
    actor_a_prob_each_round = np.transpose(np.array([agent.log for agent in agents]))
    plot_results(actor_a_prob_each_round, [str(agent) for agent in agents], path, 'player_action_probabilities',
                 ylabel='P(Cooperation)')
    planning_a_prob_each_round = np.array(planning_agent.get_log())
    # fear_and_greed_each_round = calc_fear_and_greed(planning_a_prob_each_round, env.fear, env.greed)
    plot_results(planning_a_prob_each_round, ['(D,D)', '(D,C)', '(C,D)', '(C,C)'], path, 'planning_action',
                 ylabel='a_p')
    # plot_results(fear_and_greed_each_round, ['Fear', 'Greed'], path, 'modified_fear_and_greed', ylabel='Fear/Greed')


def calc_fear_and_greed(data, base_fear, base_greed):
    assert (data.shape[1] == 2)
    assert (data.shape[2] == 2)
    fear = data[:, 0, 0] - data[:, 1, 0] + base_fear
    greed = data[:, 0, 1] - data[:, 1, 1] + base_greed
    return np.stack([fear, greed], axis=1)


if __name__ == "__main__":
    FEAR = 1
    GREED = 1
    env = Matrix_Game(fear=FEAR, greed=GREED)
    agents = create_population(env, N_PLAYERS, use_simple_agents=True)
    run_game_and_plot_results(env, agents, with_redistribution=False, max_reward_strength=3,
                              action_flip_prob=0, cost_param=0.0002, value_fn_variant='exact')
