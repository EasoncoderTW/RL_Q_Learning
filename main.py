import time
import pandas as pd
import numpy as np

# 參數設定
N_STATES = 7
ACTION = ['L','R']

# 環境
class ENV(object):
    ## 初始化
    def __init__(self):
        self.size = N_STATES
        self.status = 0
        self.rewards = [0,0,0,0,0,0,1]
    ## 重置
    def reset(self):
        self.status = 0
    ## 更新
    def update(self,action): # L/R
        if action == 'L' and self.status > 0:
            self.status -= 1
        if action == 'R' and self.status < (self.size-1):
            self.status += 1
    ## 回饋
    def get_feedback(self):
        return self.rewards[self.status]

    ## 顯示
    def __str__(self):
        """ --@----o """ # 樣式
        s = ''
        for i in range(self.size): # 由左至右
            if i == self.status:
                s += '@'
            elif self.rewards[i] > 0: # 有獎勵
                s += 'o'
            else:
                s += '-'
        return s


# 強化學習策略(Q - Learning)
class RL(object):
    ## 初始化
    def __init__(self,n_states,action,Lambda=0.9, gamma = 0.9,lr = 0.1):
        self.n_states = n_states
        self.action = action
        self.gamma = gamma
        self.Lambda = Lambda
        self.learning_rate = lr
        self.Q_table = None
        self.create_Q_table()
    ## 建立 Q 表
    def create_Q_table(self):
        self.Q_table = pd.DataFrame(
            np.zeros((self.n_states, len(self.action))),
            columns=self.action,
        )
    ## 決策
    def choose_action(self, state):
        state_action = self.Q_table.iloc[state,:]

        if (np.random.uniform() > 0.9) or (state_action.all() == 0):
            # 隨機動作
            action = np.random.choice(self.action)
        else:
            # 參考 Q 表 做決策
            action = self.action[state_action.argmax()]

        return action # str 'L'/'R'
    ## 學習 ( 更新 Q 表)
    def learn(self, r, s, a, s_n, is_terminate = False):
        q_predict = self.Q_table.loc[s,a]
        if not is_terminate:
            q_target = r + self.Lambda * self.Q_table.iloc[s_n,:].max()
        else:
            q_target = r
        # 修正
        self.Q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)

FRESH_TIME = 0.1
MAX_EPISODEs = 20

# 主程式
def main():
    # 物件 = 類別()
    env = ENV()  # 建立實體環境
    Robot = RL(N_STATES,ACTION) # 建立機器人(策略 Q Learning)
    print(env,end='')   # 輸出現在環境的狀態


    for i in range(MAX_EPISODEs):
        step_count = 0

        while True:
            state = env.status # 從環境中讀取狀態
            action = Robot.choose_action(state) # 讓機器人根據現在的狀態選擇一個動作
            env.update(action)
            feedback = env.get_feedback()
            state_next = env.status
            Robot.learn(feedback,state,action,state_next)
            step_count += 1
            print('\r', env, end='', flush=True)

            if env.status == (N_STATES-1):
                break
            else:
                time.sleep(FRESH_TIME)

        print("  Total Step:",step_count)
        print(Robot.Q_table)
        input("For Pause:")
        env.reset()
        if  i > 10:
            env.rewards[3] = 2


if __name__ == '__main__':
    main()
