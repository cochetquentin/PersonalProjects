import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataPreprocessor():
    """
    A class for preprocessing data specifically for a machine learning model. 
    It includes methods for converting video duration to seconds, 
    preprocessing numeric and categorical features, and transforming data.
    """

    def __init__(self) -> None:
        """
        Initializes the DataPreprocessor, setting up the columns for one-hot encoding 
        and standard scaling, and creating a ColumnTransformer for these operations.
        """

        self.dummies_cols = ["categoryId", "defaultLanguage", "definition"]
        self.num_cols = ["duration", "viewCount", "likeCount", "commentCount"]
        self.numeric_preprocessed = None
        self.transformer = ColumnTransformer([
                ("dummies", OneHotEncoder(handle_unknown="ignore"), self.dummies_cols),
                ("scaler", StandardScaler(), self.num_cols)
            ], remainder="passthrough", sparse_threshold=0)

    @staticmethod
    def convert_duration(duration:str) -> int:
        """
        Converts a duration string in YouTube format (e.g., 'PT1H2M3S') to seconds.

        Parameters:
        duration (str): The duration string to be converted.

        Returns:
        int: The duration in seconds.
        """

        duration = duration[2:]
        hours = 0
        minutes = 0
        seconds = 0
        if "H" in duration:
            hours = int(duration.split("H")[0])
            duration = duration.split("H")[1]
        if "M" in duration:
            minutes = int(duration.split("M")[0])
            duration = duration.split("M")[1]
        if "S" in duration:
            seconds = int(duration.split("S")[0])
        return hours*3600 + minutes*60 + seconds
    

    def keep_fill_columns(data:pd.DataFrame) -> pd.DataFrame:
        """
        Prepares and cleans the data by keeping specific columns, filling missing values,
        and converting data types.

        Parameters:
        data (pd.DataFrame): The DataFrame to be processed.

        Returns:
        pd.DataFrame: A tuple containing the processed features (X) and target variable (y).
        """

        data = data.copy(deep=True)
        keep_cat = ["categoryId", "defaultLanguage", "duration", "definition", "viewCount", "likeCount", "commentCount", "liked"]
        data = data[keep_cat]

        data.loc[~data.duration.str.startswith("PT"), "duration"] = "PT0S"
        data.loc[:, "duration"] = data["duration"].apply(DataPreprocessor.convert_duration)
        data.loc[:, "duration"] = data["duration"].astype(int)

        data.loc[:, "categoryId"] = data["categoryId"].astype(str)
        data.loc[:, "liked"] = data.loc[:, "liked"].astype(int)
        data.loc[:, "defaultLanguage"] = data["defaultLanguage"].fillna("fr")

        columns = ["viewCount", "likeCount", "commentCount"]
        for col in columns:
            data.loc[:, col] = data[col].fillna(0)
            data.loc[:, col] = data[col].astype(int)

        X, y = data.drop("liked", axis=1), data["liked"]
        return X, y
    

    def fit_transform(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Fits the transformers to the data and transforms it.

        Parameters:
        data (pd.DataFrame): The DataFrame to be processed and transformed.

        Returns:
        pd.DataFrame: The transformed features (X) and target variable (y).
        """

        X, y = DataPreprocessor.keep_fill_columns(data)
        X = self.transformer.fit_transform(X)
        X = pd.DataFrame(X, columns=self.transformer.get_feature_names_out())
        return X, y
    

    def transform(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data using already fitted transformers.

        Parameters:
        data (pd.DataFrame): The DataFrame to be transformed.

        Returns:
        pd.DataFrame: The transformed features (X) and target variable (y).
        """
        
        X, y = DataPreprocessor.keep_fill_columns(data)
        X = self.transformer.transform(X)
        X = pd.DataFrame(X, columns=self.transformer.get_feature_names_out())
        return X, y