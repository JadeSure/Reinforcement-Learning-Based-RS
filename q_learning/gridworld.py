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
import turtle
import numpy as np

# turtle tutorial : https://docs.python.org/3.3/library/turtle.html


def GridWorld(gridmap=None, is_slippery=False):
    if gridmap is None:
        gridmap = ['SFFF', 'FHFH', 'FFFH', 'HFFG']
        # desc means grid map like 4*4
    env = gym.make("FrozenLake-v0", desc=gridmap, is_slippery=False)
    env = FrozenLakeWapper(env)
    return env



class FrozenLakeWapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        # shape[0] means the dimensionality of rows
        self.max_y = env.desc.shape[0]
        # shape[1] means the dimensionality of cols
        self.max_x = env.desc.shape[1]
        self.t = None
        self.unit = 50

    def draw_box(self, x, y, fillcolor='', line_color='gray'):
        self.t.up()
        self.t.goto(x * self.unit, y * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()
        for _ in range(4):
            self.t.forward(self.unit)
            self.t.right(90)
        self.t.end_fill()

    def move_player(self, x, y):
        self.t.up()
        self.t.setheading(90)
        self.t.fillcolor('red')
        self.t.goto((x + 0.5) * self.unit, (y + 0.5) * self.unit)

    def render(self):
        if self.t == None:
            self.t = turtle.Turtle()
            self.wn = turtle.Screen()
            self.wn.setup(self.unit * self.max_x + 100,
                          self.unit * self.max_y + 100)
            self.wn.setworldcoordinates(0, 0, self.unit * self.max_x,
                                        self.unit * self.max_y)
            self.t.shape('circle')
            self.t.width(2)
            self.t.speed(0)
            self.t.color('gray')
            for i in range(self.desc.shape[0]):
                for j in range(self.desc.shape[1]):
                    x = j
                    y = self.max_y - 1 - i
                    if self.desc[i][j] == b'S':  # Start
                        self.draw_box(x, y, 'white')
                    elif self.desc[i][j] == b'F':  # Frozen ice
                        self.draw_box(x, y, 'white')
                    elif self.desc[i][j] == b'G':  # Goal
                        self.draw_box(x, y, 'yellow')
                    elif self.desc[i][j] == b'H':  # Hole
                        self.draw_box(x, y, 'black')
                    else:
                        self.draw_box(x, y, 'white')
            self.t.shape('turtle')

        x_pos = self.s % self.max_x
        y_pos = self.max_y - 1 - int(self.s / self.max_x)
        self.move_player(x_pos, y_pos)


class CliffWalkingWapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # gym.Wrapper.__init__(self, env)
        self.t = None
        self.unit = 50 # drawing pixel unit
        self.max_x = 12 # max border length
        self.max_y = 4 # max y border length

    # draw x line
    def draw_x_line(self, y, x0, x1, color='black'):
        assert x1 > x0
        self.t.color(color)
        # the angle of inclination of x
        self.t.setheading(0)
        self.t.up()
        self.t.goto(x0, y)
        self.t.down()
        self.t.forward(x1 - x0)

    def draw_y_line(self, x, y0, y1, color='gray'):
        assert y1 > y0
        self.t.color(color)
        self.t.setheading(90)
        self.t.up()
        self.t.goto(x, y0)
        self.t.down()
        self.t.forward(y1 - y0)

    def draw_box(self, x, y, fillcolor='', line_color='gray'):
        self.t.up()
        self.t.goto(x * self.unit, y * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()

        # draw ractanglar
        for i in range(4):
            self.t.forward(self.unit)
            self.t.right(90)
        self.t.end_fill()

    def move_player(self, x, y):
        self.t.up()
        self.t.setheading(90)
        # control the color of turtle
        self.t.fillcolor('pink')
        # in order to make turtle jump into the box
        self.t.goto((x+0.5 ) * self.unit, (y+0.5 ) * self.unit)

    def render(self):
        if self.t == None:
            self.t = turtle.Turtle()

            self.wn = turtle.Screen()
            self.wn.setup(self.unit * self.max_x + 100,
                          self.unit * self.max_y + 100)
            self.wn.setworldcoordinates(0, 0, self.unit * self.max_x,
                                        self.unit * self.max_y)
            # the draw point of the gridworld
            self.t.shape('circle')
            self.t.width(2)
            self.t.speed(0)
            self.t.color('pink')

            for _ in range(2):
                self.t.forward(self.max_x * self.unit)
                self.t.left(90)
                self.t.forward(self.max_y * self.unit)
                self.t.left(90)

            for i in range(1, self.max_y):
                self.draw_x_line(
                    y=i * self.unit, x0=0, x1=self.max_x * self.unit)

            for i in range(1, self.max_x):
                self.draw_y_line(
                    x=i * self.unit, y0=0, y1=self.max_y * self.unit)

            # draw box from x=1 to len(x) - 1
            for i in range(1, self.max_x - 1):
                self.draw_box(i, 0, 'black')

            # draw the last box to yellow
            self.draw_box(self.max_x - 1, 0, 'yellow')
            self.t.shape('turtle')

        x_pos = self.s % self.max_x
        y_pos = self.max_y - 1 - int(self.s / self.max_x)
        self.move_player(x_pos, y_pos)


if __name__ == '__main__':
    # env1：FrozenLake, is_slippery=False
    # 0 left, 1 down, 2 right, 3 up
    # env = gym.make("FrozenLake-v0", is_slippery=False)
    # env = FrozenLakeWapper(env)

    # env2：CliffWalking
    # env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    # env = CliffWalkingWapper(env)

    # env3：customized gridworld
    #  S == Start; F == Floor, H == Hole, G == Goal，H == Hole

    n_rows = 20
    n_cols = 20

    states_type = ['F', 'N']

    gridmap = [states_type[0] * n_cols] * n_rows


    # gridmap =[
    #            'SFFFFFFF',
    #            'FHFFFFFF',
    #            'FFFFFFFF',
    #            'HFGFFFFF' ]
    env = GridWorld(gridmap)

    env.reset()
    for step in range(10):
        action = np.random.randint(0, 4)
        obs, reward, done, info = env.step(action)
        print('step {}: action {}, obs {}, reward {}, done {}, info {}'.format(\
                step, action, obs, reward, done, info))
        env.render() # 渲染一帧图像
