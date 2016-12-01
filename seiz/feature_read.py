import pickle
import os
import pandas as pd


class FeatureReader:
    def __init__(self, feature_train_dir, feature_test_dir):
        self.feature_train_dir = feature_train_dir
        self.feature_test_dir = feature_test_dir

    def read_train_dataset(self, patients):
        """
        reads features to dataframe,
        #cls column is column with a seizure class, other columns
        are feature columns
        :param patients: list of patients to read features of
        :return: pandas data frame
        """
        result_df = pd.DataFrame()
        for p in patients:
            train_feature_file = os.path.join(self.feature_train_dir, "{}.pcl".format(p))
            with open(train_feature_file, 'rb') as f:
                features_data = pickle.load(f)
                df = pd.DataFrame(features_data)
                result_df = result_df.append(df)

        result_df['#cls'] = result_df['#cls'].astype('category')
        return result_df

    def read_test_dataset(self, patients):
        result_df = pd.DataFrame()
        for p in patients:
            test_feature_file = os.path.join(self.feature_test_dir, "{}.pcl".format(p))
            with open(test_feature_file, 'rb') as f:
                features_data = pickle.load(f)
                df = pd.DataFrame(features_data)
                result_df = result_df.append(df)
        return result_df

