import time
import pandas as pd
import numpy as np
import pygame as pg
import matplotlib.pyplot as plt


# 環境
'''MAP_SIZE = [12,8]
MAP = [
    [ 0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0],
    [ 0,  0, -1, -1, -1,  0,  0,  0, -1,  0, -1,  0],
    [ 0,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0],
    [-1,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0, -1,  0, -1, -1,  0, -1,  0, -1, -1,  0],
    [ 0,  0,  0,  0, -1,  5, -1, -1,  0, -1,  0,  0],
    [-1, -1,  0,  0, -1,  0,  0,  0, -1,  0,  0, -1],
    [ 0,  0,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0]
]'''

MAP_SIZE = [8,6]
MAP = [
    [ 0,  0,  0,  0,  0,  0, -1,  0],
    [ 0,  0, -1, -1, -1,  0,  0,  0],
    [ 0,  0, -1,  2, -1,  0, -1,  0],
    [-1,  0, -1,  0,  0,  0, -1, 10],
    [ 0,  0, -1,  0, -1,  0,  0,  0],
    [ 0,  0,  0,  0, -1,  5, -1,  0],
]
class env(object):
    def __init__(self, map_size):
        self.map_size = map_size
        self.__status = [0,0]  # 所在的位置
        self.__reward = 0 # 回饋
        # 設定視窗
        pg.init()
        self.block_size = 100
        self.gap = 50
        self.width = map_size[0]*self.block_size+self.gap*2
        self.height = map_size[1]*self.block_size+self.gap*2 # 畫面寬和高
        self.screen = pg.display.set_mode((self.width, self.height))  # 依設定顯示視窗
        pg.display.set_caption("Maze")  # 設定程式標題

    def show(self, q_table):
        self.screen.fill((0, 0, 0)) # all black
        font = pg.font.SysFont("simhei", 20)
        for x in range(self.map_size[0]):
            for y in range(self.map_size[1]):
                rect = pg.Rect(self.gap+x*self.block_size,
                               self.gap+y*self.block_size,
                               self.block_size-2,
                               self.block_size-2)
                if [x,y] == self.__status:
                    rect_color = (0,0,255)
                elif MAP[y][x] < 0:
                    rect_color = (0,0,0)
                elif MAP[y][x] == 0:
                    rect_color = (255,255,255)
                else:
                    rect_color = (127, 0, 127)
                pg.draw.rect(self.screen, rect_color, rect)

        # Q table value
        for x in range(self.map_size[0]):
            for y in range(self.map_size[1]):
                q_value = q_table.iloc[y*self.map_size[0]+x,:]
                act = ['U','D','L','R']
                position_x = [self.block_size/2*0.8,self.block_size/2*0.8,0,self.block_size*0.8]
                position_y = [0,self.block_size*0.8,self.block_size/2*0.8,self.block_size/2*0.8]
                for a,x_, y_ in zip(act, position_x, position_y):
                    if q_value[a] > 0:
                        text_color = (0, 255, 0)
                    elif q_value[a] < 0:
                        text_color = (255, 0, 0)
                    else:
                        text_color = (255, 255, 255)
                    text = font.render("{:.1f}".format(q_value[a]), True, (0,0,0), text_color)
                    self.screen.blit(text, (self.gap+x*self.block_size+x_, self.gap+y*self.block_size+y_))
        # Updates
        pg.display.update()

    def update(self, move):
        status_old = self.__status[:]
        bias = {
            'L': [-1, 0],
            'R': [ 1, 0],
            'U': [ 0,-1],
            'D': [ 0, 1],
        }
        if move in bias.keys():
            self.__status[0] += bias[move][0]
            self.__status[1] += bias[move][1]

        # hit the side wall
        if self.__status[0] < 0 or self.__status[1] < 0:
            self.__status = status_old[:]
            self.__reward = -1
        elif self.__status[0] >= self.map_size[0] or self.__status[1] >= self.map_size[1]:
            self.__status = status_old[:]
            self.__reward = -1
        else:
            self.__reward = MAP[self.__status[1]][self.__status[0]]
            if self.__reward < 0: # hit the inside wall
                self.__status = status_old[:]

    def reset(self):
        pg.quit()
        self.__status = [0,0]
        self.__reward = 0
        pg.init()
        self.screen = pg.display.set_mode((self.width, self.height))  # 依設定顯示視窗
        pg.display.set_caption("Maze")  # 設定程式標題

    def get_feedback(self):
        return self.__reward

    def get_status(self):
        return self.__status[:]

    def __del__(self):
        pg.quit() # stop pygame

# 機器學習
ACTION = ['L', 'R', 'U', 'D']
GAMMA = 0.9  # greedy policy
ALPHA = 0.1  # learning rate
LAMBDA = 0.9  # discount factor
MAX_EPISODEs = 30  # maximunm episodes
FRESH_TIME = 0.002


class RL(object):
    def __init__(self,
                 map_size,
                 action,
                 Gamma = GAMMA,
                 Alpha = ALPHA,
                 Lambda = LAMBDA):
        self.action = action
        self.map_size = map_size
        self.Gamma = Gamma
        self.Alpha = Alpha
        self.Lambda = Lambda
        self.table = None
        self.build_q_table(self.map_size[0]*self.map_size[1], ACTION) # flatten (2D -> 1D)

    def build_q_table(self, n_states, action):  # 動作參考表格(獎勵參考)
        self.table = pd.DataFrame(
            np.zeros((n_states, len(action))),
            columns=action,
        )

    def choose_action(self, state):
        state = state[1]*self.map_size[0] + state[0] # flatten (2D -> 1D)
        # How to choose an action
        state_action = self.table.iloc[state,:]

        # 隨機動作
        if (np.random.uniform() > self.Gamma) or (state_action.all() == 0):
            action_name = np.random.choice(self.action)
        else:  # act greedy
            action_name = self.action[state_action.argmax()]
        return action_name

    def update_q_table(self, R, S_N, S, A):
        #print(R,S_N,S,A)
        S_N = S_N[1]*self.map_size[0] + S_N[0]  # flatten (2D -> 1D)
        S = S[1]*self.map_size[0] + S[0]        # flatten (2D -> 1D)
        q_predict = self.table.loc[S, A]
        if S_N != self.map_size:
            q_target = R + self.Lambda * self.table.iloc[S_N, :].max()
        else:
            q_target = R
        self.table.loc[S, A] += self.Alpha * (q_target - q_predict)

def generation(gen):
    ENV = env(MAP_SIZE)  # E 物件(object) - env 類別(class)
    Robot = RL(MAP_SIZE, ACTION)
    print(ENV.get_status())

    Step_History = []
    for i in range(MAX_EPISODEs):
        ENV.reset()
        print(ENV.get_status())
        step_counter = 0
        ENV.show(q_table=Robot.table)
        print(ENV.get_status())
        print('Episode = {:2d}/{:2d}'.format(i,MAX_EPISODEs),'Step = {:2d}'.format(step_counter),end = '\n',flush=True)

        while True:
            state = ENV.get_status()
            action = Robot.choose_action(state[:])
            ENV.update(action)
            step_counter += 1
            ENV.show(q_table=Robot.table)
            print('Episode = {:2d}/{:2d}'.format(i,MAX_EPISODEs),'Step = {:2d}'.format(step_counter),end = '\n',flush=True)
            feedback = ENV.get_feedback()
            state_next = ENV.get_status()
            #print(state, action, state_next, feedback)
            Robot.update_q_table(feedback,state_next,state,action)
            if MAP[state_next[1]][state_next[0]] > 0:
                break
            else:
                time.sleep(FRESH_TIME)

        Step_History.append(step_counter)
        #print(Robot.table)

    print(Step_History)
    pg.image.save(ENV.screen, "./generation_screenshot_{:02d}.jpeg".format(gen))
    return  Step_History

def main():
    plt.figure(figsize=(8,8))
    for g in range(1,21):
        his = generation(gen=g)
        plt.plot(range(1,MAX_EPISODEs+1),his,label = "gen {:d}".format(g))
    plt.title("History of RL in 2D Maze")
    plt.xlabel("Episode")
    plt.ylabel("Step")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("./History.png")
    plt.show()

if __name__ == '__main__':
    main()
