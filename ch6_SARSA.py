import random
import numpy as np
from ch6_MCControl import GridWorld

class QAgent:
    def __init__(self):
        self.q_table = np.zeros((5,7,4)) # 마찬가지로 Q 테이블을 0으로 초기화
        self.eps = 0.9

    def select_action(self, s):
        x, y = s
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0, 3)
        else:
            action_val = self.q_table[x,y,:]
            action = np.argmax(action_val)
        return action

    def update_table(self, transition):
        s, a, r, s_prime = transition
        x, y = s
        next_x, next_y = s_prime
        a_prime = self.select_action(s_prime)

        self.q_table[x,y,a] = \
            self.q_table[x,y,a] \
            + 0.1 *(r + self.q_table[next_x,next_y,a_prime] - self.q_table[x,y,a])
    
    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)

    def show_table(self):
        q_list = self.q_table.tolist()
        data = np.zeros((5,7))
        for row_idx in range(len(q_list)):
            row = q_list[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(data)

def main():
    env = GridWorld()
    agent = QAgent()

    for n_epi in range(1000):
        done=False

        s = env.reset()
        while not done:
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            agent.update_table((s,a,r,s_prime))
            s = s_prime
        agent.anneal_eps()
    agent.show_table()

if __name__ == '__main__':
    main()