import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors

class YoutubeAPI():
    """
    A class to interact with the YouTube API, providing methods to retrieve various data like
    video details, subscriptions, uploaded videos by channel, and liked videos.
    """
    ###############################################################
    #################### STACTIC METHODS ##########################
    ###############################################################
    @staticmethod
    def channelId_to_UploadPlaylistId(channelId:str) -> str:
        """
        Converts a YouTube channel ID to its corresponding upload playlist ID.

        Parameters:
        channelId (str): The YouTube channel ID.

        Returns:
        str: The upload playlist ID corresponding to the given channel ID.
        """
        channelId = list(channelId)
        channelId[1] = "U"
        return "".join(channelId)
    
    @staticmethod
    def transform_video_data(video_data) -> dict:
        """
        Transforms raw video data from YouTube API to a more structured dictionary.

        Parameters:
        video_data (dict): The raw video data from YouTube API.

        Returns:
        dict: A dictionary containing structured video data.
        """
        videoId = video_data.get("snippet").get("thumbnails").get("high").get("url").split("/")[4]
        return {
            "videoId" : videoId,
            "title": video_data.get("snippet").get("title"),
            "publishedAt": video_data.get("snippet").get("publishedAt"),
            "channelId": video_data.get("snippet").get("channelId"),
            "description": video_data.get("snippet").get("description"),
            "thumbnails": video_data.get("snippet").get("thumbnails").get("high").get("url"),
            "channelTitle": video_data.get("snippet").get("channelTitle"),
            "tags": video_data.get("snippet").get("tags"),
            "categoryId": video_data.get("snippet").get("categoryId"),
            "defaultLanguage": video_data.get("snippet").get("defaultLanguage"),
            "duration": video_data.get("contentDetails").get("duration"),
            "definition": video_data.get("contentDetails").get("definition"),
            "viewCount": video_data.get("statistics").get("viewCount"),
            "likeCount": video_data.get("statistics").get("likeCount"),
            "commentCount": video_data.get("statistics").get("commentCount")
        }


    ###############################################################
    ###################### CLASS METHODS ##########################
    ###############################################################

    def __init__(self, client_secrets_path:str) -> None:
        """
        Initializes the YoutubeAPI object with the necessary credentials.

        Parameters:
        client_secrets_path (str): Path to the client secrets file.
        """
        self.api_service_name = "youtube"
        self.api_version = "v3"
        self.client_secrets_file = client_secrets_path
        self.get_client()


    def get_credentials(self) -> None:
        """
        Retrieves and sets the credentials for accessing the YouTube API.
        """
        scopes = [
            "https://www.googleapis.com/auth/youtube.readonly",
        ]
        flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
            self.client_secrets_file, scopes)
        self.credentials = flow.run_local_server(port=0)


    def get_client(self):
        """
        Creates and sets the YouTube API client using the credentials.
        """
        self.get_credentials()
        self.client = googleapiclient.discovery.build(
            self.api_service_name, self.api_version, credentials=self.credentials)
        

    def get_subscriptions(self) -> list:
        """
        Retrieves the list of subscriptions for the authenticated user.

        Returns:
        list: A list of tuples, each containing the channel ID and channel title of a subscription.
        """
        nextPageToken = None
        subs = []

        while True:
            request = self.client.subscriptions().list(
                part="snippet",
                mine=True,
                maxResults=50,
                pageToken=nextPageToken
            )
            response = request.execute()
            for item in response['items']:
                channelId = item['snippet']['resourceId']['channelId']
                channelTitle = item['snippet']['title']
                subs.append((channelId, channelTitle))

            if 'nextPageToken' in response:
                nextPageToken = response['nextPageToken']
            else:
                break

        return subs
    

    def get_uploadVideoIds_from_channelId(self, channelId:str, nb_videos:int=50) -> list:
        """
        Retrieves the video IDs of the uploaded videos for a given channel ID.

        Parameters:
        channelId (str): The YouTube channel ID.
        nb_videos (int, optional): The number of videos to retrieve. Default is 50.

        Returns:
        list: A list of video IDs for the uploaded videos.
        """
        playlistId = YoutubeAPI.channelId_to_UploadPlaylistId(channelId)

        nextPageToken = None
        videosIds = []
        L = nb_videos//51

        for i in range(L + 1):
            request = self.client.playlistItems().list(
                part="snippet",
                playlistId=playlistId,
                maxResults=nb_videos-L*50 if i == L else 50,
                pageToken=nextPageToken
            )
            response = request.execute()
            videosIds += [video["snippet"]["resourceId"]["videoId"] for video in response['items']]

            if 'nextPageToken' in response:
                nextPageToken = response['nextPageToken']
            else:
                break

        return videosIds
    

    def get_videoDatas_from_videoIds(self, videoIds:list) -> list:
        """
        Retrieves video data for a list of video IDs.

        Parameters:
        videoIds (list): A list of video IDs.

        Returns:
        list: A list of dictionaries, each containing structured data of a video.
        """
        request = self.client.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(videoIds)
        )
        response = request.execute()
        return [YoutubeAPI.transform_video_data(video) for video in response["items"]]


    def get_videoDatas_from_channelId(self, channelId:str, nb_videos:int=50) -> list:
        """
        Retrieves video data for all videos uploaded by a specific channel ID.

        Parameters:
        channelId (str): The YouTube channel ID.
        nb_videos (int, optional): The number of videos to retrieve. Default is 50.

        Returns:
        list: A list of dictionaries, each containing structured data of a video.
        """
        videoIds = self.get_uploadVideoIds_from_channelId(channelId, nb_videos)
        return self.get_videoDatas_from_videoIds(videoIds)
    

    def get_videoDatas_from_subscriptions(self, nb_videos:int=50) -> list:
        """
        Retrieves video data for videos from channels the user is subscribed to.

        Parameters:
        nb_videos (int, optional): The number of videos to retrieve from each subscription. Default is 50.

        Returns:
        list: A list of video data from the subscribed channels.
        """
        subs = self.get_subscriptions()
        videoDatas = []
        for channelId, channelTitle in subs:
            videoDatas += self.get_videoDatas_from_channelId(channelId, nb_videos)
        return videoDatas
    

    def get_my_likedPlaylistIds(self) -> str:
        """
        Retrieves the playlist ID of the authenticated user's liked videos.

        Returns:
        str: The playlist ID of the user's liked videos.
        """
        request = self.client.channels().list(
            part="contentDetails",
            mine=True
        )
        response = request.execute()
        return response['items'][0]['contentDetails']['relatedPlaylists']['likes']


    def get_my_liked_videoIds(self, nb_videos:int=50) -> list:
        """
        Retrieves the video IDs of the liked videos of the authenticated user.

        Parameters:
        nb_videos (int, optional): The number of liked videos to retrieve. Default is 50.

        Returns:
        list: A list of video IDs of the user's liked videos.
        """
        playlistId = self.get_my_likedPlaylistIds()

        nextPageToken = None
        videosIds = []
        L = nb_videos//51

        for i in range(L + 1):
            request = self.client.playlistItems().list(
                part="snippet",
                playlistId=playlistId,
                maxResults=nb_videos-L*50 if i == L else 50,
                pageToken=nextPageToken
            )
            response = request.execute()
            videosIds += [video["snippet"]["resourceId"]["videoId"] for video in response['items']]

            if 'nextPageToken' in response:
                nextPageToken = response['nextPageToken']
            else:
                break

        return videosIds
    

    def get_homePage_videoIds(self) -> list:
        """
        Didn't find how to get the home page video IDs for the authenticated user.
        """
        pass