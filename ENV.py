import numpy as np


L = 100
b = 0.3
R = 1
S = -b
T = 1 + b
P = 0
C = [1, 0]
C = np.asarray(C)
D = [0, 0]
D = np.asarray(D)


class PDG:
    def __init__(self):
        self.action_space = np.asarray([C, D])
        self.action_space_n = len(self.action_space)
        self.observation_space = np.asarray([C, D])

    @staticmethod
    def neighbour(i, j):  # 定义博弈发起者的四个邻居
        L_N_0 = j - 1
        R_N_1 = j + 1
        T_N_2 = i - 1
        B_N_3 = i + 1
        if i == 0:
            T_N_2 = L-1
        elif i == L-1:
            B_N_3 = 0
        if j == 0:
            L_N_0 = L-1
        elif j == L-1:
            R_N_1 = 0
        return L_N_0, R_N_1, T_N_2, B_N_3

    @staticmethod
    def Payoff_matrix(x, y):  # 定义收益矩阵
        if x == 1 and y == 1:
            p1 = R
            p2 = R
        elif x == 1 and y == 0:
            p1 = S
            p2 = T
        elif x == 0 and y == 1:
            p1 = T
            p2 = S
        else:
            p1 = P
            p2 = P
        return p1, p2

    def I_reward(self, x, x1, x2, x3, x4):
        p01, p0 = self.Payoff_matrix(x, x1)
        p02, p1 = self.Payoff_matrix(x, x2)
        p03, p2 = self.Payoff_matrix(x, x3)
        p04, p3 = self.Payoff_matrix(x, x4)
        I_reward_A = (p01+p02+p03+p04)/4  # 直接求平均值
        I_reward_Ma = max(p01, p02, p03, p04)  # 取最大值
        I_reward_Mi = min(p01, p02, p03, p04)  # 取最小值
        I_reward_A_ = ((p01+p02+p03+p04)-max(p01, p02, p03, p04)-min(p01, p02, p03, p04))/2  # 间接取平均值
        return I_reward_A, I_reward_Ma, I_reward_Mi, I_reward_A_, p0, p1, p2, p3

    def step(self, action, action0, action1, action2, action3):
        action = self.action_space[action]
        action0 = self.action_space[action0]
        action1 = self.action_space[action1]
        action2 = self.action_space[action2]
        action3 = self.action_space[action3]
        next_state = action
        next_state0 = action0
        next_state1 = action1
        next_state2 = action2
        next_state3 = action3
        I_reward_A, I_reward_Ma, I_reward_Mi, I_reward_A_, p0, p1, p2, p3 = self.I_reward(action[0], action0[0], action1[0], action2[0], action3[0])
        if action[0] == 1:
            reward = I_reward_Ma
        else:
            reward = I_reward_Mi
        # reward = I_reward_A
        # reward = I_reward_Ma
        # reward = I_reward_Mi
        # reward = I_reward_A_

        done = False
        return next_state, next_state0, next_state1, next_state2, next_state3, reward, p0, p1, p2, p3, done
