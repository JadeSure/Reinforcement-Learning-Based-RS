#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-

import gym
from gridworld import FrozenLakeWapper
from lesson2.q_learning.agent import QLearningAgent
import time

from lesson2.q_learning.RL_based_biclustering import *


def run_episode(env, agent, render=False):
    # recorder how many steps for each episode
    total_steps = 0
    total_reward = 0

    # reset the environment (a new episode for an environment)
    obs = env.reset()

    agent.states_action_set(biclusters_rows, biclusters_cols)
    # print(agent.action_set)

    while True:
        # select an new action with obs
        action = agent.sample(obs)

        stop_point = False

        if action == -1:
            stop_point = True
            action = 0

        # interact with environment
        next_obs, reward, done, _ = env.step(action)

        reward = agent.cal_rewards(obs,next_obs, biclusters_cols)

        done = stop_point
        # training Q-learning
        agent.learn(obs, action, reward, next_obs, done)

        # save the value of previous observation
        obs = next_obs
        total_reward += reward
        # count the total training steps
        total_steps += 1
        if render:
            # render a new frame
            env.render()
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0

    obs = env.reset()
    agent.states_action_set(biclusters_rows, biclusters_cols)

    while True:
        # based on Q-table
        action = agent.sample(obs, model = 'test')
        stop_point = False

        if action == -1:
            stop_point = True
            action = 0

        next_obs, _, done, _ = env.step(action)

        done = stop_point
        reward = agent.cal_rewards(obs, next_obs, biclusters_cols)
        total_reward += reward
        obs = next_obs

        time.sleep(0.5)
        env.render()
        if done:
            print('test reward = %.1f' % (total_reward))
            break


def main():
    # env = gym.make("FrozenLake-v0", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up
    # env = FrozenLakeWapper(env)
    # modify here to 20*20
    n_rows = biclusters_rows
    n_cols = biclusters_cols

    # F : flat
    # S : starting point
    states_type = ['F', 'S']
    gridmap = [states_type[0] * n_cols] * n_rows

    # TODO: set the start point with 'S'
    newstr = list(gridmap[1])
    newstr[1] = states_type[1]
    gridmap[1] = ''.join(newstr)

    env = gym.make("FrozenLake-v0",desc = gridmap, is_slippery=False)  # 0 up, 1 right, 2 down, 3 left
    env = FrozenLakeWapper(env)

    agent = QLearningAgent(
        obs_n= biclusters_rows * biclusters_cols,             # env.observation_space.n,
        act_n= 4,               # env.action_space.n,
        learning_rate=0.1,
        gamma=0.2,
        e_greed=0.1)

    is_render = False
    for episode in range(10000):
        ep_reward, ep_steps = run_episode(env, agent, is_render)
        print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps,
                                                          ep_reward))

        # show results with each 20 episode with rendering
        if episode % 20 == 0:
            is_render = True
        else:
            is_render = False

    # finished training, check the results of algorithm
    test_episode(env, agent)


if __name__ == "__main__":

    main()








