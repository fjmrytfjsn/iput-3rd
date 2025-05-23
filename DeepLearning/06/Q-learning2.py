from collections import defaultdict
import numpy as np

class MyWorld:
    def __init__(self):
        self.action_space = [0, 1, 2]  #選択出来る行動は3種類
        self.action_meaning = {        #順に、左、直進、右
            0: "LEFT",
            1: "STRAIGHT",
            2: "RIGHT",
        }
        self.reward_map = np.array(
            [[1.0, 0, 0, 0],          #左端に得点があるx4
             [1.0, 0, 0, 0],
             [1.0, 0, 0, 0],
             [1.0, 0, 0, 0],
             [1.0, -1.0, -1.0, -1.0], #左端に得点がありそれ以外は穴
             [0, 1.0, 0, 0],          #左から2番目に得点があるx4
             [0, 1.0, 0, 0],
             [0, 1.0, 0, 0],
             [0, 1.0, 0, 0],
             [-1.0, 1.0, -1.0, -1.0], #左から2番目に得点がありそれ以外は穴
             [0, 0, 1.0, 0],          #以下同様
             [0, 0, 1.0, 0],
             [0, 0, 1.0, 0],
             [0, 0, 1.0, 0],
             [-1.0, -1.0, 1.0, -1.0],
             [0, 0, 0, 1.0],
             [0, 0, 0, 1.0],
             [0, 0, 0, 1.0],
             [0, 0, 0, 1.0],
             [-1.0, -1.0, -1.0, 1.0],
            ]
        )
        self.start_state = (0,0)      #  [0]は現在のマップのID、[1]は現在のx座標
                                      #  初期状態は一番上の状態で初期位置は左端
        self.agent_state = self.start_state

    @property
    def height(self):
        return len(self.reward_map)    # 高さだが、マップの種類の数

    @property
    def width(self):
        return len(self.reward_map[0])  # 幅なのでx座標の最大値-1

    @property
    def shape(self):
        return self.reward_map.shape

    def actions(self):
        return self.action_space

#     def states(self):
#         for h in range(self.height):
#             for w in range(self.width):
#                 yield (h, w)

    def next_state(self, state, action):
        action_move_map = [-1, 0, 1]    # 行動IDと実際の移動方向のテーブル
        move = action_move_map[action]  # 変換
        nx = state[1]+move              # とりあえずx方向に移動する

        if nx < 0 or nx >= self.width:  # マップをはみ出ていたら戻す
            nx = state[1]
        if (state[0]%5)==4:              # 穴のあるマップの次はランダムで得点の位置が変わる
            ns=np.random.choice(range(4))*5
        else:
            ns=state[0]+1               # それ以外は前に進む（＝穴に近付く）
        return (ns,nx)                  # タプルにして返す

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]  # 報酬＝なし(0)または得点(1)または穴(-1)

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)     # 現在の状態と行動から次の状態を得る
        reward = self.reward(state, action, next_state) # 次の状態から報酬を得る（現在の状態と行動は使っていない）
        done = (reward == -1.0)                         # 穴に落ちたら終了
        self.agent_state = next_state                   # 次の状態に遷移
        return next_state, reward, done

    def render_q(self, q=None, print_value=False):      # Qテーブルを可視化
        for i in range(self.height):
            print(f"{i:3d}",end=": ")
            for j in range(self.width):
                qs = [q[(i,j), a] for a in range(len(self.action_space))]
                if np.argmax(qs)==0:
                    action="←"
                    if print_value:
                        print(f'\033[31m{qs[0]:5.2f}\033[0m,{qs[1]:5.2f},{qs[2]:5.2f}',end=" ")
                elif np.argmax(qs)==1:
                    action="↓"
                    if(print_value):
                        print(f'{qs[0]:5.2f},\033[31m{qs[1]:5.2f}\033[0m,{qs[2]:5.2f}',end=" ")
                elif np.argmax(qs)==2:
                    action="→"
                    if print_value:
                        print(f'{qs[0]:5.2f},{qs[1]:5.2f},\033[31m{qs[2]:5.2f}\033[0m',end=" ")
                else:
                    action="??"
                print(action,end=" ")
            print("")

#     def render_q(self, q=None, print_value=True):      # Qテーブルを可視化(数値も表示)
#         for i in range(self.height):
#             for j in range(self.width):
#                 qs = [q[(i,j), a] for a in range(len(self.action_space))]
#                 if(np.argmax(qs)==0):
#                     action="←"
#                 elif(np.argmax(qs)==1):
#                     action="↓"
#                 elif(np.argmax(qs)==2):
#                     action="→"
#                 else:
#                     action="??"
#                 print(f"({i},{j}): ",action,f'{qs[0]:5.2f},{qs[1]:5.2f},{qs[2]:5.2f}')

    # 幅の変更に追従するように変更
    def printstate(self,state,reward): # 現在の状態と報酬を表示
        out=["#"]
        for x in self.reward_map[state[0]]:
            if(x==-1):
                out.append("x")
            elif(x==1):
                out.append("+")
            else:
                out.append(".")
        out.append("#")

        if(out[state[1]+1]=="+"):
            out[state[1]+1]="O"
        elif(out[state[1]+1]=="x"):
            out[state[1]+1]="X"
        else:
            out[state[1]+1]="o"
        print(" ".join(out),reward)
        if(reward<0):
            print("GAME OVER") # 穴に落ちていたら終了


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 3
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state, best=False):
        if (best==False) and (np.random.rand() < self.epsilon):     # 学習中は確率eでランダム行動
            return np.random.choice(self.action_size)
        else:
            qs = [self.Q[state, a] for a in range(self.action_size)]
            return np.argmax(qs)                                     # それ以外は現状の最適行動

    def update(self, state, action, reward, next_state, done):        # 学習
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)                              # 次の状態からの最適行動の評価値

        target = reward + self.gamma * next_q_max                  # 今回の報酬＋次の状態の評価値
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha # 現在の自身の評価値を加味して更新


env = MyWorld()
agent = QLearningAgent()
max_step = 200

episodes = 500  # 学習上限（500回ゲームオーバーになるまで学習）
for episode in range(episodes+1):
    state = env.reset()
    step=0
    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        if done:
            break
        if step>max_step:
            break
        state = next_state
        step += 1
    if episode%50==0:             # 50回に1回、試しに本気でやってみる
        env.render_q(agent.Q,True)
        print(episode)
        state = env.reset()
        step=0
        print(f"{step+1:>3d}",end=" ")
        env.printstate(state,0.0)
        while True:
            step+=1
            action = agent.get_action(state,best=True)  # best=Trueでランダム行動を抑止
            next_state, reward, done = env.step(action)
            print(f"{step + 1:>3d}", end=" ")
            env.printstate(next_state,reward)
            if done:
                break
            if step+1>=max_step:  # max_step無事なら終了（上限を決めないと無限に終わらない可能性有り）
                print("G O A L")
                break
            state = next_state

env.render_q(agent.Q)
print("done")
