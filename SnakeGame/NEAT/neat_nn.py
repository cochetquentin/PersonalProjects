import neat
import pickle
import numpy as np
import os
from snake import Snake
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

class NeatSnakeTrainer():
    def __init__(self, config_path, checkpoint_value=0, height=800, width=800, grid_size=40, display=False, plot=False, snake_speed=200):
        self.config_path = config_path
        self.checkpoint_value = checkpoint_value
        self.config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    self.config_path)
        self.height = height
        self.width = width
        self.grid_size = grid_size
        self.display = display
        self.plot = plot
        self.snake_speed = snake_speed
        self.best_score = 0

    def plot_stats(self):
        generation = range(len(self.stats.generation_statistics))
        best_fitness = [c.fitness for c in self.stats.most_fit_genomes]
        avg_fitness = self.stats.get_fitness_mean()

        plt.close()
        plt.figure()

        plt.plot(generation, best_fitness, "b-", label="Best Fitness")
        plt.plot(generation, avg_fitness, "r-", label="Average Fitness")

        plt.xlabel("Generations")
        plt.ylabel("Fitness", color="b")

        plt.title("Fitness through generations")
        plt.legend()
        plt.pause(0.001)
        plt.show(block=False)

    def train_ai(self):
        game = Snake(self.height, self.width, self.grid_size)
        game.init_game()

        last_go_through = 0
        last_length = 0

        while game.running:
            if self.display:
                game.display()
                game.CLOCK.tick(game.SNAKE_SPEED*self.snake_speed)

            state = game.get_state()
            output = self.net.activate(state)
            choice = np.argmax(output)
            game.make_step(choice)          

            if game.snake_len == last_length:
                if game.nb_cells_go_through - last_go_through > 300:
                    game.running = False
            else:
                last_go_through = game.nb_cells_go_through
                last_length = game.snake_len

        if game.snake_len - game.starting_len > self.best_score:
            self.best_score = game.snake_len - game.starting_len
        game.quitGame()
        return game.snake_len - game.starting_len

    def eval_genome(self, genome_tuple):
        genome_id, genome = genome_tuple
        self.net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        score = self.train_ai()
        return (genome_id, score)

    def eval_genomes(self, genomes, config):
        with ProcessPoolExecutor() as executor:
            results = dict(tqdm(executor.map(self.eval_genome, genomes), total=len(genomes)))
        for genome_id, genome in genomes:
            genome.fitness = results[genome_id]
        if self.plot:
            self.plot_stats()


    def run_neat(self, nb_gen):
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    self.config_path)

        if self.checkpoint_value == 0:
            population = neat.Population(config)
        else:
            population = neat.Checkpointer.restore_checkpoint(f"checkpoint/neat-checkpoint{self.checkpoint_value}")

        population.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        population.add_reporter(self.stats)
        checkpoint = neat.Checkpointer(50, time_interval_seconds=None, filename_prefix="checkpoint/neat-checkpoint")
        population.add_reporter(checkpoint)
        winner = population.run(self.eval_genomes, int(nb_gen - self.checkpoint_value))

        with open("best.pickle", "wb") as f:
            pickle.dump(winner, f)


    def test_ai(self):
        game = Snake(self.height, self.width, self.grid_size)
        game.init_game()

        while game.running:
            game.display()
            game.CLOCK.tick(game.SNAKE_SPEED*2)

            state = game.get_state()
            output = self.net.activate(state)
            choice = np.argmax(output)
            game.make_step(choice)          

        if game.snake_len - game.starting_len > self.best_score:
            self.best_score = game.snake_len - game.starting_len
        game.quitGame()
        return game.snake_len - game.starting_len

    def eval_neat(self):
        with open("best.pickle", "rb") as f:
            genome = pickle.load(f)
        self.net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        print(self.test_ai())


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    
    trainer = NeatSnakeTrainer(config_path, checkpoint_value=0, height=800, width=800, grid_size=80, display=False, plot=False, snake_speed=200)
    trainer.run_neat(10**4)
    trainer.eval_neat()
