"""
data processing
simply call the data_prep_task function

"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_prep_task(name: str, test_size=0.2):
    """
    Function to perform data pre-processing.

    Input:
        name (str): {'cloud', 'wash'}
        test_size

    Return value: X_train, X_test, y_train, y_test
    """
    d_prep = DataPreProcessing(name)
    if name == 'cloud':
        d_prep.label_name = 'expert label'
        d_prep.drop_features(drop_cols=['y', 'x'])
        # d_prep.standardize()
        print('Data pre-processing is finished.')
        return d_prep.split(test_size=test_size)

    elif name == 'wash':
        d_prep.label_name = 'whz'
        d_prep.drop_features(threshold=0.5)

        d_prep.categorical.impute()
        # print(d_prep.categorical.indicator.dtypes)
        d_prep.disc_numerical.impute()
        # print(d_prep.disc_numerical.indicator.dtypes)
        d_prep.disc_numerical.disc_to_cont()
        d_prep.cont_numerical.impute()
        # print(d_prep.cont_numerical.indicator.dtypes)

        d_prep.standarize()
        print('Data pre-processing is finished.')
        return d_prep.split(test_size=test_size)

    else:
        raise NotImplementedError

class DataPreProcessing:
    """
    Attributes:
        name (str): {'cloud', 'wash'}
        family (str): {"Binomial", "Gaussian"}
        df (DataFrame)
        label_name (str)
        categorical (CategoricalFeatures)
        disc_numerical (DiscreteNumericalFeatures)
        cont_numerical (ContinuousNumericalFeatures)
    """

    def __init__(self, name: str):
        self.df, self.family = load_dataframe(name)
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

    def missing_values(self):
        return self.df.isnull().sum()

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


def load_dataframe(name: str):
    if name == 'wash':
        df = pd.read_csv('data/wash_data/wash_data.csv', index_col=0)
        family = 'Gaussian'
        return df, family
    elif name == 'cloud':
        df = pd.read_csv('data/cloud_data/image2.txt', sep=' ', header=None,
                         names=['y', 'x', 'expert label',
                                'NDAI', 'SD', 'CORR', 'DF', 'CF', 'BF', 'AF', 'AN'])
        family = 'Binomial'
        return df, family
    else:
        raise NotImplementedError


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
