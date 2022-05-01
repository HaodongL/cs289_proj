"""
data processing
a) drop features if majority values are missing
b) impute missingness with
   median (for continuous features) and
   mode (for discrete features) and create an indicator variable for each imputed feature
c) one hot code

"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreProcessing:
    """
    Attributes:
        path (str)
        family (str): {"Binomial", "Gaussian"}
        df (DataFrame)
        label_name (str)
        categorical (CategoricalFeatures)
        disc_numerical (DiscreteNumericalFeatures)
        cont_numerical (ContinuousNumericalFeatures)
    """

    def __init__(self, path: str, family: str):
        self.path = path
        self.family = family
        self.df = load_dataframe(path)
        self.label_name = None
        self.categorical = CategoricalFeatures(self)
        self.cont_numerical = ContinuousNumericalFeatures(self)
        self.disc_numerical = DiscreteNumericalFeatures(self)


    def get_X(self):
        return self.df[self.df.columns.difference([self.label_name])]

    def get_y(self):
        return self.df[self.label_name]

    def get_numerical(self):
        return [col for col in list(self.df.columns)
                if (self.df[col].dtype in ['float64', 'int64', 'int32']) and self.label_name != col]

    def standarize(self):
        self.df[self.get_numerical()] = pd.DataFrame(StandardScaler().fit_transform(self.df[self.get_numerical()]))
        print("Dataset is standardized.")

    def split(self, test_size=0.2, random_state=0):
        self.X = self.get_X().values
        self.y = self.get_y().values
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def drop_features(self, threshold=.8, drop_cols=[]):
        """
        Drop
        features in drop_cols
        or features whose majority values are missing (>=threshold)
        """
        if not drop_cols:
            self.df = self.df.loc[:, self.df.isnull().mean() < threshold]
        else:
            self.df = self.df.drop(drop_cols, axis=1)


def load_dataframe(path):
    return pd.read_csv(path, index_col=0)


class ContinuousNumericalFeatures:
    def __init__(self, data_obj: DataPreProcessing):
        self.data_obj = data_obj
        self.indicator = None

    def get_cont_num_features(self):
        return [col for col in list(self.data_obj.df.columns)
                if self.data_obj.df[col].dtype == 'float64']

    def impute(self, strategy='median'):
        features = self.get_cont_num_features()
        self.indicator = self.data_obj.df[features].isna()
        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        self.data_obj.df[features] = imputer.fit_transform(self.data_obj.df[features])
        print("Continuous numerical features imputed successfully.")


class DiscreteNumericalFeatures:
    def __init__(self, data_obj: DataPreProcessing):
        self.data_obj = data_obj
        self.indicator = None

    def get_disc_num_features(self):
        return [col for col in list(self.data_obj.df.columns)
                if self.data_obj.df[col].dtype in ['int64', 'int32']]

    def impute(self, strategy='most_frequent'):
        features = self.get_disc_num_features()
        self.indicator = self.data_obj.df[features].isna()
        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        self.data_obj.df[features] = imputer.fit_transform(self.data_obj.df[features])
        print("Discrete numerical features imputed successfully.")

    def disc_to_cont(self):
        features = self.get_disc_num_features()
        self.data_obj.df[features] = self.data_obj.df[features].astype(float)
        print("Int values turned into float type successfully.")


class CategoricalFeatures:
    def __init__(self, data_obj: DataPreProcessing):
        self.data_obj = data_obj
        self.indicator = None

    def get_categorical_features(self):
        return [col for col in list(self.data_obj.df.columns)
                if self.data_obj.df[col].dtype == 'object' and self.data_obj.label_name != col]

    def get_mode(self, col):
        return self.data_obj.df[col].mode().values[0]

    def impute(self):
        features = self.get_categorical_features()
        self.indicator = self.data_obj.df[features].isna()
        for col in list(features):
            mode = self.get_mode(col)
            self.data_obj.df[col].fillna(mode, inplace=True)
        print("Categorical features imputed successfully.")

    def one_hot_code(self):
        encoded_data = pd.get_dummies(self.data_obj.df.drop(self.data_obj.label_name, axis=1) )
        self.data_obj.df = pd.concat([encoded_data, self.data_obj[self.data_obj.label_name]], axis=1)
