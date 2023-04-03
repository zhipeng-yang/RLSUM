import numpy as np
import pandas as pd
from RLSUM.Algorithm import DQN
from RLSUM.ENV import PDG
from numpy import random


def main():
    # 初始化参数，智能体环境
    env = PDG()
    agent = DQN(env)
    Reward = []
    Episode = []
    Agent_i = []
    Agent_p = []
    Step = []
    P_Reward = []
    State = []
    P_State = []
    C_P = []
    P_P = []
    T_C = []
    T_P = []
    PD = np.zeros((L, L))
    PD_r = np.zeros((L, L))

    for episode in range(EPISODE):
        for step in range(STEP):
            # 初始化环境
            i = random.randint(0, L)
            j = random.randint(0, L)
            s0 = [PD[i, j], 0]
            L_N_0, R_N_1, T_N_2, B_N_3 = env.neighbour(i, j)
            s01 = [PD[i, L_N_0], 0]
            s02 = [PD[i, R_N_1], 0]
            s03 = [PD[T_N_2, j], 0]
            s04 = [PD[B_N_3, j], 0]
            state = np.array(s0, dtype=np.float32)
            state1 = np.array(s01, dtype=np.float32)
            state2 = np.array(s02, dtype=np.float32)
            state3 = np.array(s03, dtype=np.float32)
            state4 = np.array(s04, dtype=np.float32)

            action = agent.e_greedy_action(state)  # 调用epsilon_greedy算法选择动作

            action1 = agent.e_greedy_action(state1)
            action2 = agent.e_greedy_action(state2)
            action3 = agent.e_greedy_action(state3)
            action4 = agent.e_greedy_action(state4)
            next_state, next_state1, next_state2, next_state3, next_state4, reward, p1, p2, p3, p4, done = env.step(
                action, action1,
                action2, action3,
                action4)  # 执行当前动作获得所有转换数据
            # agent.perceive(state, action, reward, next_state, done)  # 调用perceive函数存储所有转换数据
            state = next_state  # 更新状态
            state1 = next_state1
            state2 = next_state2
            state3 = next_state3
            state4 = next_state4
            PD[i, j] = state[0]
            PD[i, L_N_0] = state1[0]
            PD[i, R_N_1] = state2[0]
            PD[T_N_2, j] = state3[0]
            PD[B_N_3, j] = state4[0]
            PD_r[i, j] = reward
            PD_r[i, L_N_0] = p1
            PD_r[i, R_N_1] = p2
            PD_r[T_N_2, j] = p3
            PD_r[B_N_3, j] = p4
            # print(f'Episode: {episode:<4}'
            #       f'L: {i:<5}'
            #       f'TIME_STEP: {step:<4}'
            #       f'Return: {reward:<5.1f}')
            c_p = np.sum(PD)/(L*L)
            p_p = np.sum(PD_r)/(L*L)

            Episode.append(episode)
            Step.append(step)
            Agent_i.append([i, j])
            Agent_p.append([[i, L_N_0], [i, R_N_1], [T_N_2, j], [B_N_3, j]])
            State.append(state[0])
            P_State.append([state1[0], state2[0], state3[0], state4[0]])
            Reward.append(reward)
            P_Reward.append([p1, p2, p3, p4])
            C_P.append(c_p)
            P_P.append(p_p)
            T_C.append(np.sum(PD))
            T_P.append(np.sum(PD_r))

            pd_S_data = pd.DataFrame(PD)
            pd_R_data = pd.DataFrame(PD_r)
            pd_S_data.to_csv('' + Path + '' + ENV + '_' + str(episode) + '_' + str(step) + '_s.csv')
            pd_R_data.to_csv('' + Path + '' + ENV + '_' + str(episode) + '_' + str(step) + '_r.csv')
            if 0 == step % params_interval:
                agent.update_target_params()

    pd_data = pd.DataFrame({
        'EPISODE': Episode,
        'Timestep': Step,
        'Agent_i': Agent_i,
        'Agent_p': Agent_p,
        'State': State,
        'P_State': P_State,
        'Reward': Reward,
        'P_Reward': P_Reward,
        'C_P': C_P,
        'P_P': P_P,
        'T_C': T_C,
        'T_P': T_P,
    })
    pd_data.to_csv('' + Path + '' + ENV + '.csv')


if __name__ == '__main__':
    EPISODE = 10  # 迭代周期数
    STEP = 30000  # 每个周期迭代时间步
    params_interval = 500  # 目标网络参数更新频率
    L = 100  # 方格大小
    ENV = "PD"
    Path = './data/'
    main()
