import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from langdetect import detect, LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

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
        self.transformer = ColumnTransformer([
                ("dummies", OneHotEncoder(handle_unknown="ignore"), self.dummies_cols),
                ("scaler", StandardScaler(), self.num_cols)
            ], remainder="passthrough", sparse_threshold=0)
        
        self.nlp = {
            "fr": spacy.load("fr_core_news_sm"),
            "en": spacy.load("en_core_web_sm")
        }
        self.title_vectorizer = TfidfVectorizer()
        self.description_vectorizer = TfidfVectorizer()

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
    

    def preprocessing_numeric(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Prepares and cleans the data by keeping specific columns, filling missing values,
        and converting data types.

        Parameters:
        data (pd.DataFrame): The DataFrame to be processed.

        Returns:
        pd.DataFrame: A tuple containing the processed features (X) and target variable (y).
        """

        data = data.copy(deep=True)
        data = data[self.num_cols + self.dummies_cols + ["liked"]]

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
    

    def fit_transform_numeric(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Fits the transformers to the data and transforms it.

        Parameters:
        data (pd.DataFrame): The DataFrame to be processed and transformed.

        Returns:
        pd.DataFrame: The transformed features (X) and target variable (y).
        """

        X, y = self.preprocessing_numeric(data)
        X = self.transformer.fit_transform(X)
        X = pd.DataFrame(X, columns=self.transformer.get_feature_names_out())
        return X, y
    

    def transform_numeric(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data using already fitted transformers.

        Parameters:
        data (pd.DataFrame): The DataFrame to be transformed.

        Returns:
        pd.DataFrame: The transformed features (X) and target variable (y).
        """
        
        X, y = self.preprocessing_numeric(data)
        X = self.transformer.transform(X)
        X = pd.DataFrame(X, columns=self.transformer.get_feature_names_out())
        return X, y
    

    @staticmethod
    def lematize(text:str, nlp) -> str:
        """
        Lemmatizes a text using the spaCy library.
        """
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])


    def lematize_auto(self, text:str) -> str:
        """
        Lemmatizes a text using the spaCy library, automatically detecting the language.
        """
        try:
            lang = detect(text)
        except LangDetectException:
            lang = "fr"
        return DataPreprocessor.lematize(text, self.nlp.get(lang, self.nlp["fr"]))


    def preprocessing_text(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Prepares and cleans the data by keeping specific columns, filling missing values,
        and converting data types.

        Parameters:
        data (pd.DataFrame): The DataFrame to be processed.

        Returns:
        pd.DataFrame: A tuple containing the processed features (X) and target variable (y).
        """

        data = data.copy(deep=True)
        data = data[["title", "description", "liked"]]

        data.loc[:, "title"] = data["title"].fillna("")
        data.loc[:, "description"] = data["description"].fillna("")

        data.loc[:, "title"] = data["title"].apply(lambda x: DataPreprocessor.lematize_auto(self, x))
        data.loc[:, "description"] = data["description"].apply(lambda x: DataPreprocessor.lematize_auto(self, x))

        X_title, X_description, y = data["title"], data["description"], data["liked"]
        return X_title, X_description, y
    

    def fit_transform_text(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Fits the transformers to the data and transforms it.

        Parameters:
        data (pd.DataFrame): The DataFrame to be processed and transformed.

        Returns:
        pd.DataFrame: The transformed features (X) and target variable (y).
        """

        X_title, X_description, y = DataPreprocessor.preprocessing_text(self, data)
        X_title = self.title_vectorizer.fit_transform(X_title)
        X_description = self.description_vectorizer.fit_transform(X_description)
        return X_title, X_description, y
    

    def transform_text(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data using already fitted transformers.

        Parameters:
        data (pd.DataFrame): The DataFrame to be transformed.

        Returns:
        pd.DataFrame: The transformed features (X) and target variable (y).
        """

        X_title, X_description, y = DataPreprocessor.preprocessing_text(self, data)
        X_title = self.title_vectorizer.transform(X_title)
        X_description = self.description_vectorizer.transform(X_description)
        return X_title, X_description, y