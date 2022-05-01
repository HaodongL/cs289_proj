# testing the data_processing file

from data_process import DataPreProcessing
#path_b = 'data/wash_data/wash_data.csv' # tested good on wash_data
path_a = 'data/cloud_data'

d_prep = DataPreProcessing(path_b, 'Gaussian')
d_prep.label_name = 'whz'
#print('original data size', d_prep.df.shape)

d_prep.drop_features(0.5)  # 0.004, tested good
#print('data size after dropping features', d_prep.df.shape)

d_prep.categorical.impute()  # tested good
#print(d_prep.categorical.indicator.dtypes)

d_prep.disc_numerical.impute()  # tested good
#print(d_prep.disc_numerical.indicator.dtypes)

d_prep.disc_numerical.disc_to_cont()  # tested good
#print(d_prep.df.dtypes)

d_prep.cont_numerical.impute()  # tested good
#print(d_prep.cont_numerical.indicator.dtypes)

d_prep.standarize() # tested good

#print(d_prep.df.dtypes)
X_train, X_test, y_train, y_test = d_prep.split()  # tested good
print(X_train.shape)
