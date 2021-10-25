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


env = GridWorld()

env.step(2)
env.step(2)
print(env.x, env.y)
