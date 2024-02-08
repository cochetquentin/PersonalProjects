import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
from DataPreprocessor import DataPreprocessor
from os import mkdir


class NumericRecommender():
    """
    A class for creating and managing a machine learning model for numerical data recommendations.
    It includes methods for training, saving, loading the model, and making predictions.
    """

    def __init__(self, pretrained_model_path:str=None) -> None:
        """
        Initializes the NumericRecommender with an optional pretrained model.

        Parameters:
        pretrained_model_path (str, optional): Path to a pre-trained model. If provided, the model is loaded.
        """
        self.preprocessor = DataPreprocessor()

        if pretrained_model_path:
            self.model = joblib.load(pretrained_model_path)
            self.isTrained = True
        else:
            self.model = RandomForestClassifier(criterion="gini", max_features=8, n_estimators=1000, max_depth=20)
            self.isTrained = False


    def train(self, data:pd.DataFrame) -> None:
        """
        Trains the RandomForestClassifier on the provided dataset.

        Parameters:
        data (pd.DataFrame): The dataset to train the model on.
        """
        X, y = self.preprocessor.fit_transform_numeric(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
        self.model.fit(X_train, y_train.ravel())
        self.cv = cross_val_score(self.model, X_train, y_train, cv=5)
        self.score = self.model.score(X_test, y_test)
        self.isTrained = True


    def save(self, model_path:str) -> None:
        """
        Saves the trained model and transformer to the specified path.

        Parameters:
        model_path (str): Path where the model and transformer will be saved.
        """
        if not self.isTrained:
            raise Exception("Model is not trained")
        if not model_path.endswith("/"):
            model_path += "/"

        new_model_path = model_path+"NumericRecommender/"
        try:
            mkdir(new_model_path)
        except FileExistsError:
            pass

        joblib.dump(self.preprocessor.transformer, new_model_path+"transformer.joblib")
        joblib.dump(self.model, new_model_path+"model.joblib")


    def load(self, model_path:str) -> None:
        """
        Loads the model and transformer from the specified path.

        Parameters:
        model_path (str): Path from where the model and transformer are to be loaded.
        """
        self.preprocessor.transformer = joblib.load(model_path+"transformer.joblib")
        self.model = joblib.load(model_path+"model.joblib")
        self.isTrained = True


    def predict(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Makes predictions using the trained model on the given dataset.

        Parameters:
        data (pd.DataFrame): The dataset to make predictions on.

        Returns:
        pd.DataFrame: Predicted probabilities for each class.
        """
        if not self.isTrained:
            raise Exception("Model is not trained")
        X, y = self.preprocessor.transform_numeric(data)
        return self.model.predict_proba(X)
    

    def plot_feature_importance(self) -> None:
        """
        Plots the feature importance of the trained RandomForest model.
        """
        if not self.isTrained:
            raise Exception("Model is not trained")
        importance = pd.DataFrame(
            self.model.feature_importances_, index=self.model.feature_names_in_, columns=["Importance"]
        ).sort_values(by="Importance", ascending=True)
        importance.plot.barh()





import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from NeuralNetwork import CNN
from PIL import Image

class IMGReccommender():
    """
    A class for creating and managing a neural network model for image data recommendations.
    It includes methods for dataset preparation, training, saving, loading the model, and making predictions.
    """

    def __init__(self, pretrained_model_path:str=None) -> None:
        """
        Initializes the IMGRecommender with an optional pretrained neural network model.

        Parameters:
        pretrained_model_path (str, optional): Path to a pre-trained model. If provided, the model is loaded.
        """

        self.dataset = None
        self.img_size = 256
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(180),
            transforms.Resize((self.img_size, self.img_size), antialias=True)
        ])

        self.model = CNN(
            input_size_layer=3,
            nb_hidden_layers=2,
            size_hidden_layer=16,
            nb_classes=2,
            img_shape=(self.img_size, self.img_size),
        )
        if not pretrained_model_path is None:
            self.model.load(pretrained_model_path)
            self.isTrained = True
        else:
            self.isTrained = False


    def get_dataset(self, path:str) -> ImageFolder:
        """
        Loads the image dataset from the specified path.

        Parameters:
        path (str): The path to the dataset directory.
        """

        self.dataset = ImageFolder(root=path, transform=self.transformer)


    def get_train_test(self):
        """
        Splits the loaded dataset into training and testing sets.

        Returns:
        Tuple[Subset, Subset]: A tuple containing training and testing subsets.
        """

        if self.dataset is None:
            raise Exception("Dataset is not loaded")
        return torch.utils.data.random_split(self.dataset, [0.8, 0.2])
    

    def get_train_test_loader(self):
        """
        Creates DataLoaders for the training and testing dataset.

        Returns:
        Tuple[DataLoader, DataLoader]: DataLoaders for the training and testing datasets.
        """

        train_set, test_set = self.get_train_test()
        train_loader = DataLoader(dataset=self.dataset, batch_size=128, sampler=train_set.indices)
        test_loader = DataLoader(dataset=self.dataset, batch_size=1, sampler=test_set.indices)
        return train_loader, test_loader
    

    def train(self, epochs:int=5, verbose:bool=True, dataset_path:str=None):
        """
        Trains the CNN model on the loaded dataset.

        Parameters:
        epochs (int, optional): Number of epochs to train the model. Default is 5.
        verbose (bool, optional): If True, shows training progress. Default is True.
        dataset_path (str, optional): Path to the dataset if not already loaded.
        """

        if self.dataset is None:
            self.get_dataset(dataset_path)
        train_loader, test_loader = self.get_train_test_loader()
        self.model.learning(train_loader, test_loader, epochs=epochs, verbose=verbose)
        self.isTrained = True


    def predict(self, img_path:str):
        """
        Makes a prediction for a single image using the trained CNN model.

        Parameters:
        img_path (str): The path to the image file.

        Returns:
        torch.Tensor: The prediction made by the model.
        """

        if not self.isTrained:
            raise Exception("Model is not trained")
        img = self.transformer(Image.open(img_path))
        img = img.unsqueeze(0)
        return self.model.predict(img)
    

    def save(self, path:str):
        """
        Saves the trained CNN model to the specified path.

        Parameters:
        path (str): The path where the model will be saved.
        """

        if not self.isTrained:
            raise Exception("Model is not trained")
        if not path.endswith("/"):
            path += "/"
        self.model.save(path+"IMGRecommender.pt")


    def load(self, path:str):
        """
        Loads a trained CNN model from the specified path.

        Parameters:
        path (str): The path from which to load the model.
        """
        
        self.model.load(path)
        self.isTrained = True





from sklearn.naive_bayes import MultinomialNB

class TextRecommender():
    """
    A class for creating and managing a machine learning model for text data recommendations.
    It includes methods for training, saving, loading the model, and making predictions.
    """

    def __init__(self, pretrained_model_path:str=None) -> None:
        """
        Initializes the TextRecommender with an optional pretrained model.

        Parameters:
        pretrained_model_path (str, optional): Path to a pre-trained model. If provided, the model is loaded.
        """
        self.preprocessor = DataPreprocessor()
        
        self.title_model = MultinomialNB()
        self.description_model = MultinomialNB()
        self.isTrained = False

        if pretrained_model_path:
            self.load(pretrained_model_path)


    def train(self, data:pd.DataFrame) -> None:
        """
        Trains the model on the provided dataset.

        Parameters:
        data (pd.DataFrame): The dataset to train the model on.
        """
        X_title, X_description, y_description = self.preprocessor.fit_transform_text(data)
        X_title_train, X_title_test, y_title_train, y_title_test = train_test_split(X_title, y_description, test_size=.2)
        X_description_train, X_description_test, y_description_train, y_description_test = train_test_split(X_description, y_description, test_size=.2)

        self.title_model.fit(X_title_train, y_title_train)
        self.description_model.fit(X_description_train, y_description_train)
        self.title_cv = cross_val_score(self.title_model, X_title_train, y_title_train, cv=5)
        self.description_cv = cross_val_score(self.description_model, X_description_train, y_description_train, cv=5)
        self.title_score = self.title_model.score(X_title_test, y_title_test)
        self.description_score = self.description_model.score(X_description_test, y_description_test)
        self.isTrained = True


    def save(self, model_path:str) -> None:
        """
        Saves the trained model to the specified path.

        Parameters:
        model_path (str): Path where the model will be saved.
        """
        if not self.isTrained:
            raise Exception("Model is not trained")
        if not model_path.endswith("/"):
            model_path += "/"

        new_model_path = model_path+"TextRecommender/"
        try:
            mkdir(new_model_path)
        except FileExistsError:
            pass

        joblib.dump(self.preprocessor.title_vectorizer, new_model_path+"title_vectorizer.joblib")
        joblib.dump(self.preprocessor.description_vectorizer, new_model_path+"description_vectorizer.joblib")
        joblib.dump(self.title_model, new_model_path+"title_model.joblib")
        joblib.dump(self.description_model, new_model_path+"description_model.joblib")


    def load(self, model_path:str) -> None:
        """
        Loads the model from the specified path.

        Parameters:
        model_path (str): Path from where the model is to be loaded.
        """
        self.preprocessor.title_vectorizer = joblib.load(model_path+"title_vectorizer.joblib")
        self.preprocessor.description_vectorizer = joblib.load(model_path+"description_vectorizer.joblib")
        self.title_model = joblib.load(model_path+"title_model.joblib")
        self.description_model = joblib.load(model_path+"description_model.joblib")
        self.isTrained = True


    def predict(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Makes predictions using the trained model on the given dataset.

        Parameters:
        data (pd.DataFrame): The dataset to make predictions on.

        Returns:
        pd.DataFrame: Predicted probabilities for each class.
        """
        if not self.isTrained:
            raise Exception("Model is not trained")
        X_title, X_description, y = self.preprocessor.transform_text(data)
        return self.title_model.predict_proba(X_title), self.description_model.predict_proba(X_description)