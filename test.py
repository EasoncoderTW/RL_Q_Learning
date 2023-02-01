import time
import pandas as pd
import numpy as np


# 環境
class env(object):
    def __init__(self, size):
        self.size = size  # 總長度
        self.status = 0  # 所在的位置

    def __str__(self):
        s = '\r' + ('-' * self.status + '@' + '-' * (self.size - self.status - 1))
        if self.status != self.size:
            s += 'o'
        return s

    def update(self, move):
        if move == 'L':
            if self.status > 0:
                self.status -= 1
        if move == 'R':
            if self.status < self.size:
                self.status += 1

    def reset(self):
        self.status = 0

    def get_feedback(self):
        if self.status == self.size:
            return 1
        else:
            return 0


# 機器學習
N_STATE = 7
ACTION = ['L', 'R']
GAMMA = 0.9  # greedy policy
ALPHA = 0.1  # learning rate
LAMBDA = 0.9  # discount factor
MAX_EPISODEs = 13  # maximunm episodes
FRESH_TIME = 0.1


class RL(object):
    def __init__(self):
        self.table = None
        self.build_q_table(N_STATE, ACTION)

    def build_q_table(self, n_states, action):  # 動作參考表格(獎勵參考)
        self.table = pd.DataFrame(
            np.zeros((n_states, len(action))),
            columns=action,
        )

    def choose_action(self, state):
        # How to choose an action
        state_action = self.table.iloc[state,:]

        # 隨機動作
        if (np.random.uniform() > GAMMA) or (state_action.all() == 0):
            action_name = np.random.choice(ACTION)
        else:  # act greedy
            action_name = ACTION[state_action.argmax()]
        return action_name

    def update_q_table(self, R, S_N, S, A):
        #print(R,S_N,S,A)
        q_predict = self.table.loc[S, A]
        if S_N != N_STATE:
            q_target = R + LAMBDA * self.table.iloc[S_N, :].max()
        else:
            q_target = R
        self.table.loc[S, A] += ALPHA * (q_target - q_predict)

def main():
    ENV = env(N_STATE)  # E 物件(object) - env 類別(class)
    Robot = RL()
    Step_History = []
    for i in range(MAX_EPISODEs):
        ENV.reset()
        step_counter = 0
        print(ENV,'Episode = {:2d}/{:2d}'.format(i,MAX_EPISODEs),'Step = {:2d}'.format(step_counter),end = '',flush=True)

        while True:
            state = ENV.status
            action = Robot.choose_action(state)
            ENV.update(action)
            step_counter += 1
            print(ENV,'Episode = {:2d}/{:2d}'.format(i,MAX_EPISODEs),'Step = {:2d}'.format(step_counter),end = '',flush=True)
            feedback = ENV.get_feedback()
            state_next = ENV.status
            Robot.update_q_table(feedback,state_next,state,action)
            if ENV.status == N_STATE:
                break
            else:
                time.sleep(FRESH_TIME)

        Step_History.append(step_counter)
        #print(Robot.table)

    print(Step_History)


if __name__ == '__main__':
    main()
