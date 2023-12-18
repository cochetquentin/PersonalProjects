from SnakeGame import Snake
from GameGraph import Graph
import matplotlib.pyplot as plt
import numpy as np

class Agent():
    def __init__(self) -> None:
        self.best_score = 0
        self.score_history = [0]
        self.path = []

    def get_shortest_path(self, game):
        game_matrix = game.get_game_matrix()

        game_matrix[game.snake_vect[-1][0]][game.snake_vect[-1][1]] = 0
        game_matrix[game.food[0]][game.food[1]] = 0

        graph = Graph(game_matrix)
        self.path = graph.get_shortest_path(game.snake_vect[-1], game.food)
        self.shortest_path = len(self.path) > 1
    
    @staticmethod
    def check_is_path_is_valid(game, path):
        if game.direction == "RIGHT":
           return not game.snake_vect[-1][0] - path[1][0] == -1
        elif game.direction == "LEFT":
            return not game.snake_vect[-1][0] - path[1][0] == 1
        elif game.direction == "UP":
            return not game.snake_vect[-1][1] - path[1][1] == -1
        elif game.direction == "DOWN":
            return not game.snake_vect[-1][1] - path[1][1] == 1
    
    @staticmethod
    def filter_paths(game, paths):
        return [path for path in paths if Agent.check_is_path_is_valid(game, path)]
    
    def get_random_path_of_length(self, game):
        game_matrix = game.get_game_matrix()

        game_matrix[game.snake_vect[-1][0]][game.snake_vect[-1][1]] = 0
        game_matrix[game.food[0]][game.food[1]] = 0

        graph = Graph(game_matrix)
        all_paths = graph.get_path_of_length_or_less(game.snake_vect[-1], 10)
        all_paths = Agent.filter_paths(game, all_paths)

        if len(all_paths) == 0:
            self.path = []
        else:
            self.path = all_paths[np.random.randint(len(all_paths))]

    def get_longest_path(self, game):
        game_matrix = game.get_game_matrix()

        game_matrix[game.snake_vect[-1][0]][game.snake_vect[-1][1]] = 0
        game_matrix[game.food[0]][game.food[1]] = 0

        graph = Graph(game_matrix)
        path = graph.get_longest_path(game.snake_vect[-1])

        if len(path) == 0:
            self.path = []
        else:
            self.path = path
    
    @staticmethod
    def choice_from_next_coord(snake_head_x, snake_head_y, next_x, next_y):
        print(snake_head_x, snake_head_y, next_x, next_y)
        if next_x == snake_head_x - 1:
            return 2
        elif next_x == snake_head_x + 1:
            return 3
        elif next_y == snake_head_y - 1:
            return 0
        elif next_y == snake_head_y + 1:
            return 1
        else:
            return -1

    def run(self, snake_speed=200):
        game = Snake()
        game.init_game()
        self.path = []
        self.shortest_path = False

        while game.running:
            game.display()
            game.CLOCK.tick(game.SNAKE_SPEED*snake_speed)

            if not self.shortest_path:
                self.get_shortest_path(game)
            if not self.shortest_path:
                self.get_longest_path(game)
            if len(self.path) == 0:
                choice = np.random.randint(4)
            else:
                choice = Agent.choice_from_next_coord(*game.snake_vect[-1], *self.path[1])
                self.path.pop(0)
                if len(self.path) == 1:
                    self.shortest_path = False
            game.make_step(choice)
            game.isAte()

        self.score_history.append(game.snake_len - game.starting_len)
        if game.snake_len - game.starting_len > self.best_score:
            self.best_score = game.snake_len - game.starting_len

        game.quitGame()


if __name__ == "__main__":
    agent = Agent()
    for i in range(1):
        agent.run(200)
        print(f"Game {i+1} - Score: {agent.score_history[-1]} - Best score: {agent.best_score}")
    print(f"Average score: {sum(agent.score_history)/len(agent.score_history)}")
    print(f"Best score: {agent.best_score}")
    print(f"Score history: {agent.score_history}")

    plt.plot(agent.score_history)
    plt.show()