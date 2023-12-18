import torch
from torch import nn

class NN(nn.Module):
    def __init__(self, nb_intput:int, nb_output:int, nb_hidden_layer:int, size_hidden_layer:int, learning_rate:float, device:str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nb_input = nb_intput
        self.nb_output = nb_output
        self.nb_hidden_layer = nb_hidden_layer
        self.size_hidden_layer = size_hidden_layer
        self.learning_rate = learning_rate
        self.device = device

        self.model = self.create_model()
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def create_model(self):
        model = nn.Sequential(
            nn.Linear(self.nb_input, self.size_hidden_layer),
            nn.ReLU()
        )

        for _ in range(self.nb_hidden_layer):
            model.add_module('hidden_layer', nn.Linear(self.size_hidden_layer, self.size_hidden_layer))
            model.add_module('hidden_layer_activation', nn.ReLU())

        model.add_module('output_layer', nn.Linear(self.size_hidden_layer, self.nb_output))
        return model.to(self.device)
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
        return output
    
    def fit(self, x, y):
        self.train()
        output = self.forward(x)
        loss = self.loss(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def evaluate(self, x, y):
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            loss = self.loss(output, y)
        return loss.item()