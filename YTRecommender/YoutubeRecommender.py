from DataScrapper import DataScrapper
from Recommender import NumericRecommender, IMGReccommender
import pandas as pd

class YoutubeRecommender():
    """
    A comprehensive class for YouTube video recommendation which combines numeric and image-based models.
    It handles training, saving, loading, and making predictions with both types of models.
    """

    def __init__(self, client_secret_path:str, thumbnails_folder_path:str, pretrained_numeric_model_path:str=None, pretrained_img_model_path:str=None) -> None:
        """
        Initializes the YoutubeRecommender with required paths and optional pretrained model paths.

        Parameters:
        client_secret_path (str): Path to the client secret file for YouTube API.
        thumbnails_folder_path (str): Path to store and access video thumbnails.
        pretrained_numeric_model_path (str, optional): Path to a pre-trained numeric model.
        pretrained_img_model_path (str, optional): Path to a pre-trained image model.
        """

        self.client_secret_path = client_secret_path
        self.thumbnails_folder_path = thumbnails_folder_path
        self.pretrained_model_path = pretrained_numeric_model_path
        self.pretrained_img_model_path = pretrained_img_model_path

        self.scrapper = DataScrapper(client_secret_path)
        self.numericReco = NumericRecommender(pretrained_numeric_model_path)
        self.imgReco = IMGReccommender(pretrained_img_model_path)

        self.videoToPredicts = None
        self.num_predicts = None
        self.img_predicts = None
        self.predicts = None

        self.scrapper.structured_folder(self.thumbnails_folder_path)


    def train_numeric_model(self, nb_liked_videos:int=5000, nb_subs_videos:int=50, model_save_path:str=None) -> None:
        """
        Trains the numeric model using data obtained from the DataScrapper.

        Parameters:
        nb_liked_videos (int, optional): Number of liked videos to include. Default is 5000.
        nb_subs_videos (int, optional): Number of subscription videos to include. Default is 50.
        model_save_path (str, optional): Path to save the trained numeric model.
        """

        numeric_data = self.scrapper.get_numeric_data(nb_liked_videos, nb_subs_videos)
        self.numericReco.train(numeric_data)
        if not model_save_path is None:
            self.numericReco.save(model_save_path)

    
    def load_numeric_model(self, model_path:str) -> None:
        """
        Loads a pre-trained numeric model from the specified path.

        Parameters:
        model_path (str): Path from which to load the numeric model.
        """
        self.numericReco.load(model_path)


    def train_img_model(self, epoch:int=5, verbose:bool=True, model_save_path:str=None) -> None:
        """
        Trains the image model using the thumbnails obtained from the DataScrapper.

        Parameters:
        epoch (int, optional): Number of epochs to train the model. Default is 5.
        verbose (bool, optional): If True, displays training progress. Default is True.
        model_save_path (str, optional): Path to save the trained image model.
        """

        self.scrapper.get_thumbnails(self.thumbnails_folder_path)
        self.imgReco.train(epoch, verbose, self.thumbnails_folder_path+self.scrapper.train_folder)
        if not model_save_path is None:
            self.imgReco.save(model_save_path)


    def load_img_model(self, model_path:str) -> None:
        """
        Loads a pre-trained image model from the specified path.

        Parameters:
        model_path (str): Path from which to load the image model.
        """

        self.imgReco.load(model_path)


    def train_model(self, nb_liked_videos:int=5000, nb_subs_videos:int=50, epoch:int=5, verbose:bool=True,  
                    numeric_model_save_path:str=None, img_model_save_path:str=None) -> None:
        """
        Trains both numeric and image models and optionally saves them.

        Parameters:
        nb_liked_videos (int, optional): Number of liked videos to include. Default is 5000.
        nb_subs_videos (int, optional): Number of subscription videos to include. Default is 50.
        epoch (int, optional): Number of epochs to train the image model. Default is 5.
        verbose (bool, optional): If True, displays training progress. Default is True.
        numeric_model_save_path (str, optional): Path to save the trained numeric model.
        img_model_save_path (str, optional): Path to save the trained image model.
        """

        self.train_numeric_model(nb_liked_videos, nb_subs_videos, numeric_model_save_path)
        self.train_img_model(epoch, verbose, img_model_save_path)


    def load_model(self, models_folder_path:str) -> None:
        """
        Loads both numeric and image models from the specified folder path.

        Parameters:
        models_folder_path (str): The folder path where both models are stored.
        """

        if models_folder_path[-1] != "/":
            models_folder_path += "/"

        self.load_numeric_model(f"{models_folder_path}NumericRecommender/")
        self.load_img_model(f"{models_folder_path}IMGRecommender.pt")


    def get_numeric_recommendation(self) -> None:
        """
        Generates numeric recommendations based on the home page video data.
        """

        if self.videoToPredicts is None:
            self.videoToPredicts = self.scrapper.get_videoData_toPredict()
        if self.numericReco.isTrained is False:
            raise Exception("Numeric model is not trained")

        self.num_predicts = self.numericReco.predict(self.videoToPredicts)
        self.num_predicts = pd.DataFrame(self.num_predicts, columns=["num_0", "num_1"])
        self.num_predicts.index = self.videoToPredicts.index


    def get_thumbnail_recommendation(self) -> None:
        """
        Downloads thumbnails for the home page video data to be used for image model predictions.
        """

        if self.videoToPredicts is None:
            self.videoToPredicts = self.scrapper.get_videoData_toPredict()

        videosIds = self.videoToPredicts.index.tolist()
        group_videoIds = [videosIds[i:i+50] for i in range(0, len(videosIds)-1, 50)]
        for group in group_videoIds:
            self.scrapper.dl_thumbnail_from_videoIds(group, self.thumbnails_folder_path)
        

    def get_img_recommendation(self) -> None:
        """
        Generates image-based recommendations using the downloaded thumbnails.
        """

        if self.videoToPredicts is None:
            self.videoToPredicts = self.scrapper.get_videoData_toPredict()
        if self.imgReco.isTrained is False:
            raise Exception("Img model is not trained")
        
        self.scrapper.dl_thumbnail_from_videoIds(self.videoToPredicts.index.tolist(), self.thumbnails_folder_path+self.scrapper.predict_folder)

        self.img_predicts = self.videoToPredicts.apply(
            lambda x: self.imgReco.predict(f"{self.thumbnails_folder_path}{self.scrapper.predict_folder}{x.name}.jpg").tolist()[0], axis=1)
        self.img_predicts = self.img_predicts.apply(pd.Series)
        self.img_predicts.columns = ["img_0", "img_1"]


    def get_recommendations(self) -> None:
        """
        Combines both numeric and image-based predictions into a single recommendation.
        """

        if self.num_predicts is None:
            self.get_numeric_recommendation()
        if self.img_predicts is None:
            self.get_img_recommendation()

        self.predicts = self.num_predicts.merge(self.img_predicts, left_index=True, right_index=True)
        self.predicts["0"] = self.predicts.loc[:, ["num_0", "img_0"]].mean(axis=1)
        self.predicts["1"] = self.predicts.loc[:, ["num_1", "img_1"]].mean(axis=1)


    def ranking(self) -> pd.DataFrame:
        """
        Ranks the videos based on the combined recommendation scores.

        Returns:
        pd.DataFrame: The ranked list of video URLs with their corresponding recommendation scores.
        """

        if self.predicts is None:
            raise Exception("Predictions are not loaded")
        preds = self.predicts["1"]
        preds.index = "https://www.youtube.com/watch?v=" + preds.index
        return preds.round(2).sort_values(ascending=False)