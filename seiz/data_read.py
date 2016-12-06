from scipy.io import loadmat
import os
import numpy as np


def _read_mat_data(mat_file_name):
    """
    Returns dict with keys:
        * signals -- electrode signals pandas dataframe
        * sampling_rate -- number of measurements per second
    """
    try:
        mat = loadmat(mat_file_name)
    except ValueError as e:
        print("value error for mat filguppye {}".format(mat_file_name))
        return {'signals': None, 'sampling_rate': None}
    names = mat['dataStruct'].dtype.names
    n_data = {n: mat['dataStruct'][n][0, 0] for n in names}
    return {'signals': n_data['data'],
            'column_names': n_data['channelIndices'][0],
            'sampling_rate': mat['dataStruct']['iEEGsamplingRate'][0, 0][0, 0]}


class MatPatientDataReader:
    def __init__(self, patient_no, data_dir):
        self.data_dir = data_dir
        self.patient_no = patient_no

    def get_train_files(self):
        """
        returns array of tuples (patient_no, train_data_idx, seiz_stage_class, path_to_mat_file)
        for given patient number
        """
        train_dir = os.path.join(self.data_dir, "train_{}".format(self.patient_no))
        filenames = os.listdir(train_dir)
        interm = ((os.path.splitext(f)[0].split("_"), os.path.join(train_dir, f)) for f in filenames)
        return [(int(p[0][0]), int(p[0][1]), int(p[0][2]), p[1]) for p in interm]

    def get_test_files(self):
        """
        returns array of tuples (patient_no, train_data_idx, path_to_mat_file)
        for given patient number
        """
        train_dir = os.path.join(self.data_dir, "test_{}_new".format(self.patient_no))
        filenames = os.listdir(train_dir)
        interm = ((os.path.splitext(f)[0].split("_"), os.path.join(train_dir, f)) for f in filenames)
        return [(int(p[0][1]), int(p[0][2]), p[1]) for p in interm]

    def random_train_sample(self, cls):
        import random
        train = [t for t in self.get_train_files() if t[2] == cls]
        return {'cls': cls, **_read_mat_data(random.choice(train)[3])}

    def train_samples_for_cls(self, cls):
        """
        get train samples for particular seizure stage class
        """
        train = [t for t in self.get_train_files() if t[2] == cls]
        all_samples = ({'cls': t[2], **_read_mat_data(t[3])} for t in train)
        return (sample for sample in all_samples if sample['signals'] is not None), len(train)

    def train_samples(self):
        train = self.get_train_files()
        all_samples = ({'cls': t[2], **_read_mat_data(t[3])} for t in train)
        return (sample for sample in all_samples if sample['signals'] is not None), len(train)

    def test_samples(self):
        test = self.get_test_files()
        all_samples = ({'mat_name': os.path.split(t[2])[1], **_read_mat_data(t[2])} for t in test)
        return (sample for sample in all_samples if sample['signals'] is not None), len(test)
