from YoutubeAPI import YoutubeAPI
import pandas as pd
from requests import get
from os import listdir, mkdir

class DataScrapper():
    """
    A class designed for scraping and structuring data from YouTube using the YoutubeAPI. 
    It includes methods for extracting video data, downloading thumbnails, and organizing this data 
    for machine learning purposes.
    """
    
    def __init__(self, client_secret_path:str) -> None:
        """
        Initializes the DataScrapper object with YouTube API credentials and sets up folder paths 
        for storing training and prediction data.

        Parameters:
        client_secret_path (str): Path to the client secret file for YouTube API authentication.
        """

        self.yt = YoutubeAPI(client_secret_path)
        self.video_data = None

        self.train_folder = "toTrain/"
        self.predict_folder = "toPredict/"
        self.liked_folder = "liked/"
        self.disliked_folder = "disliked/"


    @staticmethod
    def videoDatas_to_df(videoDatas:dict) -> pd.DataFrame:
        """
        Converts video data into a pandas DataFrame.

        Parameters:
        videoDatas (dict): A dictionary containing video data.

        Returns:
        pd.DataFrame: A DataFrame representation of the video data.
        """

        dfs = [pd.DataFrame.from_dict(video, orient='index').T for video in videoDatas]
        return pd.concat(dfs).reset_index(drop=True)

    ####################################################################################################################
    ########################################### NUMERIC DATA ###########################################################
    ####################################################################################################################

    def get_subsVideos_df(self, nb_videos:int) -> pd.DataFrame:
        """
        Retrieves video data from the user's subscriptions and converts it into a DataFrame.

        Parameters:
        nb_videos (int): Number of videos to retrieve from each subscription.

        Returns:
        pd.DataFrame: A DataFrame containing video data from the user's subscriptions.
        """

        subs_videos = self.yt.get_videoDatas_from_subscriptions(nb_videos)
        return DataScrapper.videoDatas_to_df(subs_videos)
    

    def get_videos_dfs(self, nb_liked_videos:int=5000, nb_subs_videos:int=50) -> pd.DataFrame:
        """
        Retrieves data frames of subscribed videos and liked videos, marking them accordingly.

        Parameters:
        nb_liked_videos (int, optional): Number of liked videos to retrieve.
        nb_subs_videos (int, optional): Number of subscription videos to retrieve.

        Returns:
        pd.DataFrame: DataFrames for subscribed and liked videos.
        """

        sub_videos_df = self.get_subsVideos_df(nb_subs_videos)
        liked_videos = self.yt.get_my_liked_videoIds(nb_liked_videos)

        sub_videos_df["liked"] = sub_videos_df["videoId"].isin(liked_videos)

        liked_videos_not_scrapped_ids = [lv for lv in liked_videos if lv not in sub_videos_df["videoId"].tolist()]

        liked_videos_not_scrapped = []
        for i in range(0, len(liked_videos_not_scrapped_ids), 50):
            liked_videos_not_scrapped += self.yt.get_videoDatas_from_videoIds(liked_videos_not_scrapped_ids[i:i+50])
            
        liked_videos_df = DataScrapper.videoDatas_to_df(liked_videos_not_scrapped)
        liked_videos_df["liked"] = True

        return sub_videos_df, liked_videos_df


    def get_video_data(self, nb_liked_videos:int=5000, nb_subs_videos:int=50) -> pd.DataFrame:
        """
        Combines subscribed and liked video data into a single DataFrame.

        Parameters:
        nb_liked_videos (int, optional): Number of liked videos to retrieve.
        nb_subs_videos (int, optional): Number of subscription videos to retrieve.

        Returns:
        pd.DataFrame: A combined DataFrame of subscribed and liked video data.
        """
        if self.video_data is None:
            sub_videos_df, liked_videos_df = self.get_videos_dfs(nb_liked_videos, nb_subs_videos)
            self.video_data = pd.concat([sub_videos_df, liked_videos_df]).reset_index(drop=True)
            return self.video_data
        else:
            return self.video_data
    

    ####################################################################################################################
    ############################################### IMG DATA ###########################################################
    ####################################################################################################################

    @staticmethod
    def dl_thumbnail_from_url(url:str, path:str) -> None:
        """
        Downloads a video thumbnail from a given URL and saves it to a specified path.

        Parameters:
        url (str): URL of the video thumbnail.
        path (str): Path where the thumbnail will be saved.
        """

        response = get(url)
        try:
            with open(path, "xb") as f:
                f.write(response.content)
        except FileExistsError:
            pass


    def structured_folder(self, thumbnails_folder_path:str) -> None:
        """
        Creates a structured folder hierarchy for storing thumbnails.

        Parameters:
        thumbnails_folder_path (str): The base path for the thumbnail folders.
        """
        
        if thumbnails_folder_path[-1] != "/":
            thumbnails_folder_path += "/"

        folders = listdir(thumbnails_folder_path)
        if self.train_folder[:-1] not in folders:
            mkdir(thumbnails_folder_path+self.train_folder)
        if self.predict_folder[:-1] not in folders:
            mkdir(thumbnails_folder_path+self.predict_folder)

        folders = listdir(thumbnails_folder_path+self.train_folder)
        if self.liked_folder[:-1] not in folders:
            mkdir(thumbnails_folder_path+self.train_folder+self.liked_folder)
        if self.disliked_folder[:-1] not in folders:
            mkdir(thumbnails_folder_path+self.train_folder+self.disliked_folder)


    def get_thumbnail_url_from_videoIds(self, videoIds:list) -> list:
        """
        Retrieves thumbnail URLs for a list of video IDs.

        Parameters:
        videoIds (list): A list of video IDs.

        Returns:
        list: A list of thumbnail URLs corresponding to the given video IDs.
        """

        thb = []
        for i in range(0, len(videoIds), 50):
            data = self.yt.get_videoDatas_from_videoIds(videoIds[i:i+50])
            thb += [video["thumbnails"] for video in data]
        return thb
    

    def dl_thumbnail_from_videoIds(self, videoIds:list, path:str) -> None:
        """
        Downloads thumbnails for a list of video IDs and saves them to a specified path.

        Parameters:
        videoIds (list): A list of video IDs.
        path (str): The path where thumbnails will be saved.
        """

        urls = self.get_thumbnail_url_from_videoIds(videoIds)
        Already_dl = [f.split(".")[0] for f in listdir(path)]
        for i in range(len(urls)):
            if videoIds[i] not in Already_dl:
                DataScrapper.dl_thumbnail_from_url(urls[i], f"{path}{videoIds[i]}.jpg")


    def put_img_in_dataset_from_url(self, videoId:str, url:str, liked:int, thumbnails_folder_path:str) -> None:
        """
        Downloads a thumbnail from a URL and categorizes it into liked or disliked based on user preference.

        Parameters:
        videoId (str): The ID of the video.
        url (str): The URL of the thumbnail.
        liked (int): Indicates whether the video is liked (1) or not (0).
        thumbnails_folder_path (str): The base path for storing the thumbnail.
        """

        folder = thumbnails_folder_path+self.liked_folder if liked else thumbnails_folder_path+self.disliked_folder
        file_name = folder + videoId + ".jpg"
        DataScrapper.dl_thumbnail_from_url(url, file_name)
        

    def get_thumbnails(self, thumbnails_folder_path:str) -> None:
        """
        Downloads and organizes thumbnails for the videos present in numeric data.

        Parameters:
        thumbnails_folder_path (str): The base path for storing thumbnails.
        """

        if self.video_data is None:
            raise Exception("Numeric data is not loaded")
        
        liked_videoIds_dl = [f.split(".")[0] for f in listdir(thumbnails_folder_path+self.train_folder+self.liked_folder)]
        disliked_videoIds_dl = [f.split(".")[0] for f in listdir(thumbnails_folder_path+self.train_folder+self.disliked_folder)]
        videoIds_dl = liked_videoIds_dl + disliked_videoIds_dl
        self.video_data["dl"] = self.video_data["videoId"].isin(videoIds_dl)
        
        thumbnails = self.video_data[~self.video_data["dl"]]
        thumbnails = thumbnails.loc[:, ["videoId", "thumbnails", "liked"]]
        thumbnails.loc[:, "liked"] = thumbnails["liked"].astype(int)

        _ = thumbnails.apply(lambda row: self.put_img_in_dataset_from_url(row["videoId"], row["thumbnails"], row["liked"], thumbnails_folder_path+self.train_folder), axis=1)


    ####################################################################################################################
    ############################################### GET DATA ###########################################################
    ####################################################################################################################
        
    def get_data_for_training(self, thumbnails_folder_path:str, nb_liked_videos:int=5000, nb_subs_videos:int=50) -> None:
        """
        Prepares and structures data for training, including downloading and organizing thumbnails.

        Parameters:
        thumbnails_folder_path (str): The base path for storing thumbnails.
        nb_liked_videos (int, optional): Number of liked videos to process.
        nb_subs_videos (int, optional): Number of subscribed videos to process.
        """

        self.get_video_data(nb_liked_videos, nb_subs_videos)
        self.get_thumbnails(thumbnails_folder_path)


    def get_videoData_toPredict(self) -> pd.DataFrame:
        """
        **I wanted to get the video's homepage, but I didn't find the way to do it for the authenticated user.**

        Get the last videos from each subscription for prediction.

        Returns:
        pd.DataFrame: A DataFrame containing the datas of last videos from each subscription.
        """

        videoDatas = self.get_subsVideos_df(1)
        videoDatas["liked"] = 1
        videoDatas.index = videoDatas["videoId"]
        return videoDatas