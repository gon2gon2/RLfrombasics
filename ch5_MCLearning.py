import random
import numpy as np

class GridWorld():
    def __init__(self):
        self.x=0
        self.y=0
    
    def step(self, a):
        # 0번 액션: 왼쪽, 1번 액션: 위, 2번 액션: 오른쪽, 3번 액션: 아래쪽
        if a==0:
            self.move_left()
        elif a==1:
            self.move_up()
        elif a==2:
            self.move_right()
        elif a==3:
            self.move_down()

        reward = -1 # 보상은 항상 -1로 고정
        done = self.is_done()
        return (self.x, self.y), reward, done

    def move_right(self):
        self.y += 1  
        if self.y > 3:
            self.y = 3
      
    def move_left(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0
      
    def move_up(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0
  
    def move_down(self):
        self.x += 1
        if self.x > 3:
            self.x = 3

    def is_done(self):
        if self.x == 3 and self.y == 3:
            return True
        else :
            return False

    def get_state(self):
        return (self.x, self.y)
      
    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)

class Agent:
    def __init__(self):
        pass

    def select_action(self):
        coin = random.random()
        if coin < 0.25:
            action = 0
        elif coin < 0.5:
            action = 1
        elif coin < 0.75:
            action = 2
        else:
            action = 3
        return action

def main():
    env = GridWorld()
    agent = Agent()
    data = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] # 테이블 초기화
    gamma = 1.0
    alpha = 0.0001
    
    for k in range(50000): # 5만개의 에피소드 진행
        done = False
        history = []
        while not done: # while ~ reset이 하나의 에피소드
            action = agent.select_action()
            (x, y), reward, done = env.step(action)
            history.append((x,y,reward))
        env.reset()

        # 매 에피소드가 끝나고 바로 해당 데이터를 이용해 테이블을 업데이트
        cum_reward = 0
        for transition in history[::-1]: # 왜 뒤에서부터 할까?, 하나의 에피소드에 대한 히스토리
            x, y, reward = transition    # 뒤에서부터 계산하는 이유: 가장 마지막 행동의 리턴이 감쇠인자를 많이 먹어야 돼서
            '''
            에피소드가 끝나면 가장 마지막에 수행한 액션, 리워드부터 계산
            재귀적 계산
            G_t = G_t+1 * r + R_t+1

            '''
            data[x][y] = data[x][y] + alpha*(cum_reward-data[x][y])
            cum_reward = cum_reward + gamma*reward

    #학습이 끝난 후 데이터 출력을 위한 코드
    for row in data:
        print(row)
if __name__ == '__main__':
    main()