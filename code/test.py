# testing the data_processing file

from data_process import DataPreProcessing

d_prep = DataPreProcessing("cloud")

d_prep.label_name = 'expert label'
# d_prep.drop_features(drop_cols=['y', 'x'])
# d_prep.standardize()
print('Data pre-processing is finished.')
d_prep.y_con_to_disc()
d = d_prep.get_y()
print(d)
print(d.isnull().sum())