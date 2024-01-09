import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) class for image classification tasks. 
    It includes layers for convolution, pooling, and classification, 
    and methods for training, evaluation, saving/loading models, and plotting loss history.
    """

    def __init__(self, input_size_layer:int, nb_hidden_layers:int, size_hidden_layer:int, nb_classes:int, img_shape:tuple, device:str='cuda') -> None:
        """
        Initializes the CNN model with given parameters and sets up the layers.

        Parameters:
        input_size_layer (int): The number of input channels (e.g., 3 for RGB images).
        nb_hidden_layers (int): Number of hidden layers in the network.
        size_hidden_layer (int): The number of channels produced by the convolutional layers.
        nb_classes (int): The number of classes for classification.
        img_shape (tuple): The shape of the input images (height, width).
        device (str, optional): The device to run the model on ('cuda' or 'cpu'). Default is 'cuda'.
        """

        super(CNN, self).__init__()
        self.input_size_layer = input_size_layer
        self.nb_hidden_layer = nb_hidden_layers
        self.size_hidden_layer = size_hidden_layer
        self.nb_classes = nb_classes
        self.device = device
        
        self.train_loss_history = []
        self.train_score_history = []
        self.test_loss_history = []
        self.test_score_history = []
        
        # Compute img size after convolutions
        self.final_img_size = int((img_shape[0]/2**self.nb_hidden_layer)*(img_shape[1]/2**self.nb_hidden_layer))
        
        # Create model
        self.model = nn.Sequential()
        for i in range(self.nb_hidden_layer):
            if i == 0:
                self.model.add_module('conv'+str(i), self.create_conv_layer(self.input_size_layer, self.size_hidden_layer))
            else:
                self.model.add_module('conv'+str(i), self.create_conv_layer(self.size_hidden_layer, self.size_hidden_layer))
        self.model.add_module("flatten", nn.Flatten())
        self.model.add_module("linear", self.create_linear_layer(self.final_img_size*self.size_hidden_layer, nb_classes))
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)


    @staticmethod
    def create_conv_layer(input_size_layer:int, output_size_layer:int, kernel_size:int=3, stride:int=1, padding:int=1) -> nn.Sequential:
        """
        Creates a convolutional layer for the CNN.

        Parameters:
        input_size_layer (int): The number of input channels.
        output_size_layer (int): The number of output channels.
        kernel_size (int, optional): Size of the convolving kernel. Default is 3.
        stride (int, optional): Stride of the convolution. Default is 1.
        padding (int, optional): Zero-padding added to both sides of the input. Default is 1.

        Returns:
        nn.Sequential: A sequential container of the convolutional layer followed by ReLU and MaxPool2d.
        """

        return nn.Sequential(
            nn.Conv2d(input_size_layer, output_size_layer, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(output_size_layer, output_size_layer, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    
    @staticmethod
    def create_linear_layer(input_size_layer:int, output_size_layer:int) -> nn.Sequential:
        """
        Creates a linear layer for the CNN.

        Parameters:
        input_size_layer (int): Size of each input sample.
        output_size_layer (int): Size of each output sample.

        Returns:
        nn.Sequential: A sequential container of a linear layer followed by Softmax.
        """

        return nn.Sequential(
            nn.Linear(input_size_layer, output_size_layer),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x) -> torch.Tensor:
        """
        Defines the forward pass of the CNN.

        Parameters:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor after passing through the CNN.
        """

        return self.model(x)
    

    def predict(self, x) -> torch.Tensor:
        """
        Makes predictions using the CNN on given input data.

        Parameters:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output predictions.
        """

        x = x.to(self.device)
        with torch.inference_mode():
            return self(x)
    

    def evaluate(self, test_loader:torch.utils.data.DataLoader) -> None:
        """
        Evaluates the CNN on a given test dataset.

        Parameters:
        test_loader (torch.utils.data.DataLoader): The DataLoader containing the test dataset.
        """

        correct = 0
        total = 0
        test_loss = 0

        self.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)

                test_loss += self.criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            self.test_loss_history.append(test_loss / len(test_loader))
            self.test_score_history.append(correct / total)


    def fit(self, train_loader:torch.utils.data.DataLoader, verbose:bool=True) -> None:
        """
        Trains the CNN on a given training dataset.

        Parameters:
        train_loader (torch.utils.data.DataLoader): The DataLoader containing the training dataset.
        verbose (bool, optional): If True, shows a progress bar of the training process. Default is True.
        """

        train_loss = 0
        correct = 0
        total = 0

        self.train()
        for images, labels in tqdm(train_loader, leave=False, disable=not verbose, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self(images)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        self.train_loss_history.append(train_loss / len(train_loader))
        self.train_score_history.append(correct / total)
    

    def learning(self, train_loader:torch.utils.data.DataLoader, test_loader:torch.utils.data.DataLoader, epochs:int=10, verbose:bool=True) -> None:
        """
        Conducts the learning process over multiple epochs, including both training and evaluation.

        Parameters:
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        epochs (int, optional): Number of epochs to train for. Default is 10.
        verbose (bool, optional): If True, prints progress and results each epoch. Default is True.
        """

        for epoch in range(epochs):
            self.fit(train_loader)
            self.evaluate(test_loader)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {self.train_loss_history[-1]:.4f} | Train Score: {self.train_score_history[-1]:.4f} | Test Loss: {self.test_loss_history[-1]:.4f} | Test Score: {self.test_score_history[-1]:.4f}")




    def save(self, path:str) -> None:
        """
        Saves the CNN model to a specified path.

        Parameters:
        path (str): The path where the model will be saved.
        """

        torch.save(self.state_dict(), path)


    def load(self, path:str) -> None:
        """
        Loads a CNN model from a specified path.

        Parameters:
        path (str): The path from which to load the model.
        """

        self.load_state_dict(torch.load(path))


    def plot_loss(self) -> None:
        """
        Plots the training and test loss history of the CNN.
        """
        
        plt.figure(figsize=(10, 5))
        plt.title("Loss History")
        plt.plot(self.train_loss_history, label="train")
        plt.plot(self.test_loss_history, label="test")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()