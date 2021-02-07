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

import numpy as np
from lesson2.q_learning.RL_based_biclustering import *


class QLearningAgent(object):
    def __init__(self,
                 obs_n, # states space
                 act_n,
                 learning_rate=0.01,
                 gamma=0.9,
                 e_greed=0.1):
        self.act_n = act_n  # action space
        self.lr = learning_rate  # learning rate
        self.gamma = gamma  # the discount rate of reward
        self.epsilon = e_greed  # the percent of selection random actions
        self.Q = np.zeros((obs_n, act_n)) # Q table with states space in rows
                                          # actions in cols
        self.action_set = {}



    def states_action_set(self, rows_number, cols_number):

        check_list = []
        # set actionset for each state in 4 corner specific situation
        self.action_set[self.__reverse_version_transition(0,0, cols_number)] = [1,2]
        self.action_set[self.__reverse_version_transition(0, cols_number-1, cols_number)] = [0, 1]
        self.action_set[self.__reverse_version_transition(rows_number-1, 0, cols_number)] = [2, 3]
        self.action_set[self.__reverse_version_transition(rows_number-1, cols_number-1, cols_number)] = [0, 3]

        check_list.append(self.__reverse_version_transition(0,0, cols_number))
        check_list.append(self.__reverse_version_transition(0, cols_number-1, cols_number))
        check_list.append(self.__reverse_version_transition(rows_number-1, 0, cols_number))
        check_list.append(self.__reverse_version_transition(rows_number-1, cols_number-1, cols_number))

        # set rows limitation
        for i in range(self.__reverse_version_transition(0,1, cols_number),
                       self.__reverse_version_transition(0, cols_number-1, cols_number)):
            check_list.append(i)
            self.action_set[i] = [0,1,2]

        for j in range(self.__reverse_version_transition(rows_number-1, 1, cols_number),
                       self.__reverse_version_transition(rows_number-1, cols_number-1, cols_number)):
            check_list.append(j)
            self.action_set[j] = [0,2,3]

        # set cols limitation
        for k in range(self.__reverse_version_transition(1, 0, cols_number),
                       self.__reverse_version_transition(rows_number-1, 0, cols_number), cols_number):
            check_list.append(k)
            self.action_set[k] = [1,2,3]

        for w in range(self.__reverse_version_transition(1, cols_number-1, cols_number),
                       self.__reverse_version_transition(rows_number-1, cols_number-1, cols_number), cols_number):
            check_list.append(w)
            self.action_set[w] = [0,1,3]

        # set the rest of states to all function
        for q in range(0, rows_number*cols_number):
            if q in check_list:
                continue
            check_list.append(q)
            self.action_set[q] = [0, 1, 2, 3]

    # transform value in observation state version to row, col version
    def __version_transition(self, value, cols_number):
        row = int(value / cols_number)
        col = value % cols_number
        return row, col

    # according to the index of row and col to calculate state index
    def __reverse_version_transition(self, row, col, cols_number):
        return row * cols_number + col

    # according to observed value --> find next action with exploration
    def sample(self, obs, model = 'train'):

        if model == 'test':
            self.epsilon = 0

        # remove all possible action spaces to this state
        # eg. for 0, remove states number 1 and 5
        self.remove_relevant_actions(obs)

        # if there is no more movement space to select
        if len(self.action_set[obs]) == 0:
            action = -1
            return action

        # according to Q-table to select actions
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            action = self.predict(obs)
        else:
            # to explore an action randomly
            action_list = self.action_set[obs]
            action = np.random.choice(action_list)

        return action


    # based on oberservation value to predict next action
    def predict(self, obs):
        if len(self.action_set[obs]) == 0:
            action = -1
            return action

        action_spaces = self.action_set[obs]

        # print('currrent obs: ', obs, ' with ', action_spaces)

        # obtain a list of value from table Q
        Q_list = self.Q[obs, action_spaces]
        maxQ = np.max(Q_list)

        # if there are many actions that have same value in Q table, it will select the first one
        action_index = np.where(Q_list == maxQ)[0]
        # print('action list: ', action_index)
        # print(np.random.choice(action_index))

        action = action_spaces[np.random.choice(action_index)]
        # print('action is: ', action)

        return action


    # def judge_visited(self, next_obs):
    #     row, col = self.__version_transition(next_obs, biclusters_cols)
    #
    #     # == 0, never be visited
    #     if self.visited_state[row][col] == 0:
    #         return False
    #     else:
    #         # visited already
    #         return True

    # in the current state, remove relevant action space to this state
    def remove_relevant_actions(self, obs):

        for i in self.check_available_states(obs):

            states_num = self.__reverse_version_transition(i[0], i[1], biclusters_cols)
            # print('current obs: ', obs, 'state num', states_num)
            # print('current states number: ',states_num)
            # print(self.action_set[states_num])
            if obs - states_num > 1:
                self.action_set[states_num].remove(1)

            if obs - states_num == 1:
                self.action_set[states_num].remove(2)

            if obs - states_num == -1:
                self.action_set[states_num].remove(0)

            if obs - states_num < -1:
                # print('remove level', self.action_set[states_num])
                self.action_set[states_num].remove(3)


    # check available states in the current observation
    def check_available_states(self, obs):
        row, col = self.__version_transition(obs,biclusters_cols)
        available_states_list = []


        if row + 1 < biclusters_rows :
            available_states_list.append([row+1, col])

        if row - 1 >= 0:
            available_states_list.append([row-1, col])

        if col + 1 < biclusters_cols:
            available_states_list.append([row, col + 1])

        if col -1 >= 0:
            available_states_list.append([row, col -1])

        # print(available_states_list)

        return available_states_list


    # learning ways, updating Q table
    def learn(self, obs, action, reward, next_obs, done):
        """ off-policy
            obs: before interaction --> obs, s_t
            action: the action that has been selected in this period, a_t
            reward: the reward that obtained from this operation
            next_obs: after interaction, the obs status, s_t+1
            done: check whether episode is finished
        """
        predict_Q = self.Q[obs, action]
        if done:
            # there is no next status, just assign target_Q with reward directly
            target_Q = reward
        else:
            # Q(S,A) + a[R + λ*max_Q(S', A') - Q(S, A)]
            target_Q = reward + self.gamma * np.max(self.Q[next_obs, :])  # Q-learning
        # assign target_Q into Q table
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)

    # save Q the value of Q table into a file
    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    # read Q table from the file
    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')

    # calculate the reward between two states
    def cal_rewards(self, current_state_index, next_state_index, cols_number):
        current_row, current_col = self.__version_transition(current_state_index, cols_number)
        new_row, new_col = self.__version_transition(next_state_index, cols_number)

        # find user list in this status
        a = new_dict_biclusters[states[current_row,current_col]][0]
        b = new_dict_biclusters[states[new_row, new_row]][0]

        return self.__jaccard_sim(a,b)



    # calculate rewards based on jaccard similarity
    def __jaccard_sim(self, a, b):
        intersection = len(set(a).intersection(set(b)))
        unions = len(set(a).union(set(b)))

        return 1. * intersection/unions


if __name__ == '__main__':
    action_set = {}

    def __reverse_version_transition(row, col, cols_number):
        return row * cols_number + col

    def states_action_set(rows_number, cols_number):
        # set actionset for each state in 4 corner specific situation
        action_set[__reverse_version_transition(0, 0, cols_number)] = [1, 2]
        action_set[__reverse_version_transition(0, cols_number - 1, cols_number)] = [0, 1]
        action_set[__reverse_version_transition(rows_number - 1, 0, cols_number)] = [2, 3]
        action_set[__reverse_version_transition(rows_number - 1, cols_number - 1, cols_number)] = [0, 3]

        # set rows limitation
        for i in range(__reverse_version_transition(0, 1, cols_number),
                       __reverse_version_transition(0, cols_number - 1, cols_number)):
            action_set[i] = [0, 1, 2]

        for j in range(__reverse_version_transition(rows_number - 1, 1, cols_number),
                       __reverse_version_transition(rows_number - 1, cols_number - 1, cols_number)):
            action_set[j] = [0, 2, 3]

        # set cols limitation
        for k in range(__reverse_version_transition(1, 0, cols_number),
                       __reverse_version_transition(rows_number - 1, 0, cols_number),
                       cols_number):
            action_set[k] = [1, 2, 3]

        for w in range(__reverse_version_transition(1, cols_number - 1, cols_number),
                       __reverse_version_transition(rows_number - 1, cols_number - 1, cols_number),
                       cols_number):
            action_set[w] = [0, 1, 3]

    states_action_set(5,5)




    for i in range(0,25):
        if i in action_set:
            continue
        action_set[i] = [0,1,2,3]

    print(action_set.keys())

    print(action_set[3])
    print("===================")
    print(action_set.values())
    action_set[3].remove(0)
    action_set[3].remove(1)

    # for i in range(25):
    #     print(action_set[i])
    print(action_set[3])

    print(action_set.values())



    # print(version_transition(40,20))




