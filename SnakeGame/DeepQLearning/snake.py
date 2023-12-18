import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame as pg
import numpy as np


class Snake():
    def __init__(self, height, width, grid_size, random_seed = None) -> None:
        if random_seed != None:
            np.random.seed(random_seed)
        pg.init()

        self.SCREEN = None
        self.HEIGHT = height
        self.WIDTH = width
        self.GRID_SIZE = grid_size
        self.CLOCK = pg.time.Clock()
        self.SNAKE_SPEED = 10

        self.SNAKE_COLOR = (0, 150, 0)
        self.FOOD_COLOR = (150, 0, 0)

        self.ACTION = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.FONT = pg.font.Font('freesansbold.ttf', 32)

        self.snake_vect = []
        self.nb_cells_go_through = 0
        self.starting_len = 2
        self.snake_len = self.starting_len
        self.direction = None
        self.food = None

    def draw_snake(self):
        for x, y in self.snake_vect:
            rect = pg.Rect(self.GRID_SIZE*x, self.GRID_SIZE*y, self.GRID_SIZE, self.GRID_SIZE)
            pg.draw.rect(self.SCREEN, self.SNAKE_COLOR, rect)

    def draw_food(self):
        x, y = self.food
        rect = pg.Rect(x*self.GRID_SIZE, y*self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        pg.draw.rect(self.SCREEN, self.FOOD_COLOR, rect)

    def spawn_food(self):
        available = []
        for x in range(self.WIDTH // self.GRID_SIZE):
            for y in range(self.HEIGHT // self.GRID_SIZE):
                if (x, y) not in self.snake_vect:
                    available.append((x, y))
        self.food = (available[np.random.randint(0, len(available))])

    def isAte(self):
        if self.snake_vect[-1] == self.food:
            self.snake_len += 1
            self.spawn_food()

    def update_snake(self):
        if self.direction == "RIGHT":
            next_pos = (self.snake_vect[-1][0] + 1, self.snake_vect[-1][1])
        elif self.direction == "LEFT":
            next_pos = (self.snake_vect[-1][0] - 1, self.snake_vect[-1][1])
        elif self.direction == "UP":
            next_pos = (self.snake_vect[-1][0], self.snake_vect[-1][1] - 1)
        elif self.direction == "DOWN":
            next_pos = (self.snake_vect[-1][0], self.snake_vect[-1][1] + 1)

        if not self.collision(next_pos[0], next_pos[1]):
            self.snake_vect.append(next_pos)
            self.nb_cells_go_through += 1
        else:
            self.running = False
        if self.snake_len < len(self.snake_vect):
            self.snake_vect.pop(0)

    def move_snake(self, event):
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_LEFT and self.direction != "RIGHT":
                self.direction = "LEFT"
            elif event.key == pg.K_RIGHT and self.direction != "LEFT":
                self.direction = "RIGHT"
            elif event.key == pg.K_UP and self.direction != "DOWN":
                self.direction = "UP"
            elif event.key == pg.K_DOWN and self.direction != "UP":
                self.direction = "DOWN"

    def starting_snake_pos(self):
        nb_cells_w = self.WIDTH // self.GRID_SIZE
        nb_cells_h = self.HEIGHT // self.GRID_SIZE
        x = np.random.randint(int(nb_cells_w/4), int(nb_cells_w*3/4))
        y = np.random.randint(int(nb_cells_h/4), int(nb_cells_h*3/4))
        self.snake_vect.append((x, y))

    def collision(self, x, y):
        if (x >= self.HEIGHT // self.GRID_SIZE
            or x < 0
            or y >= self.WIDTH // self.GRID_SIZE
            or y < 0):
            return True
        elif len(set(self.snake_vect)) != len(self.snake_vect):
            return True

    def init_game(self):
        self.running = True
        self.starting_snake_pos()
        self.spawn_food()
        self.direction = np.random.choice(self.ACTION)

    def display_score(self):
        text = self.FONT.render(f"Score: {self.snake_len-self.starting_len}", True, (255, 255, 255))
        textRect = text.get_rect()
        textRect.left = 0
        self.SCREEN.blit(text, textRect)

    def make_step(self, choix:int):
        self.move_ai(choix)
        self.update_snake()
        r = self.get_reward()
        self.isAte()
        return r

    def display(self):
        if self.SCREEN is None:
            self.SCREEN = pg.display.set_mode((self.WIDTH, self.HEIGHT))
            pg.display.set_caption("Snake")
        self.SCREEN.fill((0, 0, 0))
        self.draw_snake()
        self.draw_food()
        self.display_score()
        pg.display.update()

    def quitGame(self):
        self.running = False
        pg.quit()

    def run(self):
        self.init_game()
        start_time = pg.time.get_ticks()

        while self.running:
            self.display()
            self.CLOCK.tick(self.SNAKE_SPEED)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                self.move_snake(event)
            
            self.update_snake()
            self.isAte()

        self.game_time = pg.time.get_ticks() - start_time
        pg.quit()

    def move_ai(self, choix:int):
        choix = self.ACTION[choix]
        if choix == "UP" and self.direction != "DOWN":
            self.direction = "UP"
        elif choix == "DOWN" and self.direction != "UP":
            self.direction = "DOWN"
        elif choix == "LEFT" and self.direction != "RIGHT":
            self.direction = "LEFT"
        elif choix == "RIGHT" and self.direction != "LEFT":
            self.direction = "RIGHT"
    
    def get_game_matrix(self):
        mat = np.zeros((self.HEIGHT // self.GRID_SIZE, self.WIDTH // self.GRID_SIZE))
        for x, y in self.snake_vect:
            mat[x,y] = 1
        mat[self.food[0], self.food[1]] = 2
        return mat

    def get_state(self):
        game_matrix = self.get_game_matrix()
        snake_head_x, snake_head_y = self.snake_vect[-1]

        if snake_head_y == 0:
            danger_up = 1
        else:
            cell_up = game_matrix[snake_head_x, snake_head_y - 1]
            danger_up = int((cell_up == 1))

        if snake_head_y == self.HEIGHT // self.GRID_SIZE - 1:
            danger_down = 1
        else:
            cell_down = game_matrix[snake_head_x, snake_head_y + 1]
            danger_down = int((cell_down == 1))

        if snake_head_x == 0:
            danger_left = 1
        else:
            cell_left = game_matrix[snake_head_x - 1, snake_head_y]
            danger_left = int((cell_left == 1))

        if snake_head_x == self.WIDTH // self.GRID_SIZE - 1:
            danger_right = 1
        else:
            cell_right = game_matrix[snake_head_x + 1, snake_head_y]
            danger_right = int((cell_right == 1))

        isUp = int(self.direction == "UP")
        isDown = int(self.direction == "DOWN")
        isLeft = int(self.direction == "LEFT")
        isRight = int(self.direction == "RIGHT")

        isFoodUp = int(self.food[1] < snake_head_y)
        isFoodDown = int(self.food[1] > snake_head_y)
        isFoodLeft = int(self.food[0] < snake_head_x)
        isFoodRight = int(self.food[0] > snake_head_x)

        # Body distance
        dist_body_up = self.HEIGHT // self.GRID_SIZE
        dist_body_down = self.HEIGHT // self.GRID_SIZE
        dist_body_left = self.HEIGHT // self.GRID_SIZE
        dist_body_right = self.HEIGHT // self.GRID_SIZE

        for x, y in self.snake_vect:
            if x == snake_head_x:
                if y < snake_head_y:
                    dist_body_up = snake_head_y - y
                elif y > snake_head_y:
                    dist_body_down = y - snake_head_y
            elif y == snake_head_y:
                if x < snake_head_x:
                    dist_body_left = snake_head_x - x
                elif x > snake_head_x:
                    dist_body_right = x - snake_head_x

        dist_body_up /= self.HEIGHT // self.GRID_SIZE
        dist_body_down /= self.HEIGHT // self.GRID_SIZE
        dist_body_left /= self.HEIGHT // self.GRID_SIZE
        dist_body_right /= self.HEIGHT // self.GRID_SIZE

        return (danger_up, danger_down, danger_left, danger_right, isUp, isDown, isLeft, isRight, isFoodUp, isFoodDown, isFoodLeft, isFoodRight, dist_body_up, dist_body_down, dist_body_left, dist_body_right)
    
    def get_reward(self):
        if not self.running:
            return -10
        elif self.snake_vect[-1] == self.food:
            return 10
        else:
            return 0
    
if __name__ == "__main__":
    snake = Snake(800, 800, 20)
    snake.run()