import torch
import numpy as np
import pandas as pd

from snake import Snake
from NN import NN

class DeepQLearning():
    def __init__(self, epsilon, gamma, device, nb_input, nb_output, nb_hidden_layer, size_hidden_layer, learning_rate) -> None:
        self.NN = NN(nb_input, nb_output, nb_hidden_layer, size_hidden_layer, learning_rate, device)
        self.DEVICE = device
        self.GAMMA = gamma
        self.MAX_MEMORY = 10**5
        self.BATCH_SIZE = 1000

        self.epsilon = epsilon
        self.best_score = 0
        self.score_history = [0]
        self.memory = []

    def check_memory(self):
        if len(self.memory) > self.MAX_MEMORY:
            self.memory = self.memory[-self.MAX_MEMORY:]

    def add_memory(self, state, action, reward, next_state):
        value = (state, action, reward, next_state)
        if value not in self.memory:
            self.memory.append(value)
            self.check_memory()

    def to_tensor(self, value):
        return torch.tensor(value, dtype=torch.float32, device=self.DEVICE).reshape(1, -1)
    
    def run(self, height, width, grid_size, display=False, snake_speed=200):
        game = Snake(height, width, grid_size)
        game.init_game()

        last_go_through = 0
        last_length = 0

        while game.running:
            if display:
                game.display()
                game.CLOCK.tick(game.SNAKE_SPEED*snake_speed)

            state = game.get_state()
            if np.random.rand() < self.epsilon:
                output = np.random.rand(4)
            else:
                output = self.NN.predict(self.to_tensor(state))
            choice = output.argmax().item()
                
            reward = game.make_step(choice)
            next_state = game.get_state()

            self.fit_model_short_term(state, choice, reward, next_state)
            self.add_memory(state, choice, reward, next_state)
            self.check_memory()

            if game.snake_len == last_length:
                if game.nb_cells_go_through - last_go_through > 20**3:
                    game.running = False
            else:
                last_go_through = game.nb_cells_go_through
                last_length = game.snake_len

        self.score_history.append(game.snake_len - game.starting_len)
        if game.snake_len - game.starting_len > self.best_score:
            self.best_score = game.snake_len - game.starting_len
        game.quitGame()

    def calculate_target(self, reward, next_state):
        next_state = self.to_tensor(next_state)
        return reward + self.GAMMA * self.NN.predict(next_state).max().item()

    def calculate_q_value(self, state, action, target):
        state = self.to_tensor(state)
        q_values = self.NN.predict(state)
        q_values[0][action] = target
        return q_values.reshape(1, -1).to("cpu")

    def get_batch(self):
        batch_size = min(self.BATCH_SIZE, len(self.memory))
        data = pd.DataFrame(self.memory, columns=["state", "action", "reward", "next_state"])
        batch = data.sample(batch_size, replace=False)
        batch["target"] = batch.apply(lambda x: self.calculate_target(x["reward"], x["next_state"]), axis=1)
        batch["q_value"] = batch.apply(lambda x: self.calculate_q_value(x["state"], x["action"], x["target"]), axis=1)
        return batch["state"], batch["q_value"]

    def fit_model_short_term(self, state, action, reward, next_state):
        target = self.calculate_target(reward, next_state)
        q_value = self.calculate_q_value(state, action, target)
        state = self.to_tensor(state)
        q_value = q_value.to(self.DEVICE)
        self.NN.fit(state, q_value)

    def fit_model_long_term(self):
        inputs, outputs = self.get_batch()
        inputs = inputs.apply(lambda x: self.to_tensor(x))
        inputs = torch.cat(inputs.to_list(), dim=0).to(self.DEVICE)
        outputs = torch.cat(outputs.to_list(), dim=0).to(self.DEVICE)
        self.NN.fit(inputs, outputs)

    def train(self, height, width, grid_size, nb_epoch=10, display=False, save=True, checkpoint=100, snake_speed=200):
        for k in range(nb_epoch):
            print(f"Epoch {k}: Best score: {self.best_score} | Last score: {self.score_history[-1]} | Mean score (50 last games): {np.mean(self.score_history[-50:])} | Epsilon: {self.epsilon}", end="\r")
            self.run(height, width, grid_size, display, snake_speed)
            if len(self.memory) > 0:
                self.fit_model_long_term()
                self.epsilon = max(0.001, self.epsilon * 0.99)
            if save and k % checkpoint == 0:
                self.save(f"model.pth")

    def save(self, path):
        torch.save(self.NN.state_dict(), path)

    def load(self, path):
        self.NN.load_state_dict(torch.load(path))

    def test_ia(self, height, width, grid_size, ia_speed=2):
        game = Snake(height, width, grid_size)
        game.init_game()

        while game.running:
            game.display()
            game.CLOCK.tick(game.SNAKE_SPEED*ia_speed)

            state = game.get_state()
            output = self.NN.predict(self.to_tensor(state))
            choice = output.argmax().item()

            game.make_step(choice)

        game.quitGame()
        return game.snake_len - game.starting_len