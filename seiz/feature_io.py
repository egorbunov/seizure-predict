import pickle
import os

class FeatureIO:
    def __init__(self, feature_base_dir):
        self.feature_train_dir = os.path.join(feature_base_dir, "train")
        self.feature_test_dir = os.path.join(feature_base_dir, "test")
        if not os.path.exists(self.feature_test_dir):
            os.makedirs(self.feature_test_dir)
        if not os.path.exists(self.feature_train_dir):
            os.makedirs(self.feature_train_dir)

    def read(self, patient, which="train", segment_size=None):
        """
        reads features to dataframe,
        #cls column is column with a seizure class, other columns
        are feature columns
        :param segment_size: size of the sample, into which initial sample were split
        :param which: train or test
        :param patient: patient no
        :return: pandas data frame
        """

        if which == "train":
            feat_file = self.feature_train_dir
        elif which == "test":
            feat_file = self.feature_test_dir
        else:
            raise KeyError("which may be train or test")

        if segment_size is not None:
            feat_file = os.path.join(feat_file, "{}".format(segment_size))
        train_feature_file = os.path.join(feat_file, "{}.pcl".format(patient))
        features = []
        with open(train_feature_file, 'rb') as f:
            while True:
                try:
                    features.append(pickle.load(f))
                except (EOFError, pickle.PickleError, pickle.UnpicklingError) as e:
                    break
            return features

    def write(self, features):
        """
        :param features: as they returned from feature calculator
        :return:
        """
        segment_size = features['segment_size']
        patient = features['patient']
        features_gen = features['features']
        which = features['which']

        if which == "train":
            base = self.feature_train_dir
        elif which == "test":
            base = self.feature_test_dir
        else:
            raise KeyError("which may be train or test")

        base_dir = os.path.join(base, "{}".format(segment_size))
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        feat_file = os.path.join(base_dir, "{}.pcl".format(patient))
        with open(feat_file, 'wb') as f:
            for feature in features_gen:
                pickle.dump(feature, f)
