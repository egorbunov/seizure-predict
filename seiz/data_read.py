import pandas as pd
from scipy.io import loadmat
import os


def _get_df(mat_file_name):
    try:
        mat = loadmat(mat_file_name)
    except ValueError as e:
        print("value error in mat file")
        return None
    names = mat['dataStruct'].dtype.names
    n_data = {n: mat['dataStruct'][n][0, 0] for n in names}
    return pd.DataFrame(n_data['data'], columns=n_data['channelIndices'][0])


class MatPatientDataReader:
    def __init__(self, patient_no, data_dir):
        self.data_dir = data_dir
        self.patient_no = patient_no

    def _get_train_files(self):
        """
        returns array of tuples (patient_no, train_data_idx, seiz_stage_class, path_to_mat_file)
        for given patient number
        """
        train_dir = os.path.join(self.data_dir, "train_{}".format(self.patient_no))
        filenames = os.listdir(train_dir)
        interm = ((os.path.splitext(f)[0].split("_"), os.path.join(train_dir, f)) for f in filenames)
        return [(int(p[0][0]), int(p[0][1]), int(p[0][2]), p[1]) for p in interm]

    def _get_test_files(self):
        """
        returns array of tuples (patient_no, train_data_idx, path_to_mat_file)
        for given patient number
        """
        train_dir = os.path.join(self.data_dir, "test_{}_new".format(self.patient_no))
        filenames = os.listdir(train_dir)
        interm = ((os.path.splitext(f)[0].split("_"), os.path.join(train_dir, f)) for f in filenames)
        return [(int(p[0][1]), int(p[0][2]), p[1]) for p in interm]

    def get_train_data_generator(self):
        train = self._get_train_files()
        return (df for df in ({'cls': t[2], 'df': _get_df(t[3])} for t in train) if df['df'] is not None), len(train)

    def get_test_data_generator(self):
        test = self._get_test_files()
        return (d for d in ({'mat_name': os.path.split(t[2])[1], 'df': _get_df(t[2])} for t in test)
                if d['df'] is not None), len(test)
