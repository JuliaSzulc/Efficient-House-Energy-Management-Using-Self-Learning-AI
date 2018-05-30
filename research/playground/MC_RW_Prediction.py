from collections import defaultdict

from enviroments.Path1DEnv import Path1DEnv
import random
import matplotlib.pyplot as plt

"""
This script performs Monte-Carlo Prediction of a fully-random policy for the Random Walk problem:

- first-visit only MC policy evaluation
- every-visit MC policy evaluation

then it plots the value functions given by these methods for comparison.

Additionally you can specify if the environment should be cyclic or not.

Note that this is not the incremental algorithm in which 
it updates the value function with each episode.
"""


def random_policy(state):
    return random.choice(['l', 'r'])


def create_value_fn_fig(number_of_states, first_visit_v, every_visit_v, figname='value_fn_eval_MC.png'):

    """Creates figure for the value function and saves it"""

    fig = plt.figure(figsize=(9, 3))
    ax1 = fig.add_subplot(111)

    y_first = [first_visit_v[x] for x in range(number_of_states)]
    y_every = [every_visit_v[x] for x in range(number_of_states)]

    ax1.scatter(range(number_of_states), y_first, s=2, c='b', label='first_visit')
    ax1.scatter(range(number_of_states), y_every, s=2, c='r', label='every_visit')
    plt.legend(loc='upper right')

    plt.title("MC Policy Evaluation - Random Walk")
    plt.xlabel("State")
    plt.ylabel("Value Function")
    plt.savefig(figname)


def mc_policy_eval(env, policy, num_episodes, discount, type="First"):

    value_fn = defaultdict(lambda: 0)
    episodes = [env.sample_episode(policy) for x in range(0, num_episodes)]
    print(episodes[0])
    count = defaultdict(lambda: 0)
    g_return = defaultdict(lambda: 0)

    for episode in episodes:
        visited = defaultdict(lambda: False)
        for i, timeframe in enumerate(episode):
            if type == "First" and visited[timeframe[0]]:
                continue
            state = timeframe[0]
            visited[state] = True
            count[state] += 1
            g_return[state] += sum([(discount**(t-i)) * episode[t+1][2] for t in range(i, len(episode) - 1)])

    for state in count.keys():
        # print(state, count[state], g_return[state])
        value_fn[state] = g_return[state] / count[state] if count[state] != 0 else 0

    return value_fn


if __name__ == "__main__":

    num_of_all_states = 100
    terminal_states = [3, 30, 44, 69, 73]
    path_env = Path1DEnv(True, num_of_all_states, terminal_states)

    num_of_episodes = 1000
    discount_rate = 0.9

    first_v = mc_policy_eval(path_env, random_policy, num_of_episodes, discount_rate, type="First")
    every_v = mc_policy_eval(path_env, random_policy, num_of_episodes, discount_rate, type="Every")
    create_value_fn_fig(num_of_all_states, first_v, every_v)

