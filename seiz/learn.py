from seiz.feature_read import FeatureReader
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm


class SeizureModel:
    def __init__(self, feature_train_dir, feature_test_dir, patients):
        self.feature_reader = FeatureReader(feature_train_dir, feature_test_dir)
        self.patients = patients
        self.cls_col = '#cls'
        self.mat_name_col = '#mat_name'
        self.models = {}
        self.train_data = {}
        self.test_data = {}
        self.all_patients = [1, 2, 3]
        self.feature_columns = [
            # 'sum_std',
            # 'sum_mean',
            # 'sum_perc_30',
            # 'sum_perc_70',
            # 'sum_fft_std',
            # 'sum_fft_mean',
            # 'sum_fft_perc_30',
            # 'sum_fft_perc_70'
        ]

        electrode_num = 16

        for i in range(electrode_num):
            self.feature_columns.append('{}_mean'.format(i))
            self.feature_columns.append('{}_fft_mean'.format(i))
            self.feature_columns.append('{}_perc_30'.format(i))
            self.feature_columns.append('{}_perc_70'.format(i))
            # self.feature_columns.append('{}_fft_std'.format(i))
            # self.feature_columns.append('{}_std'.format(i))

        for p in patients:
            if p not in set(self.all_patients):
                raise RuntimeError("Bad patient id")
            self.models[p] = RandomForestClassifier(
                n_estimators=2000,
                n_jobs=8
            )
            self.train_data[p] = self.feature_reader.read_train_dataset([p])

        for p in self.all_patients:
            self.test_data[p] = self.feature_reader.read_test_dataset([p])

    def do_cross_validation(self):
        score_means = {}
        for p in self.patients:
            skf = StratifiedKFold(n_splits=5)
            features = self.train_data[p].drop([self.cls_col], axis=1)
            features = features[self.feature_columns].as_matrix()
            answers = self.train_data[p][self.cls_col].as_matrix()
            scores = []
            for train_index, test_index in tqdm(skf.split(features, answers)):
                f_train, f_test = features[train_index], features[test_index]
                a_train, a_test = answers[train_index], answers[test_index]
                self.models[p].fit(f_train, a_train)
                scores.append(self.models[p].score(f_test, a_test))
            score_means[p] = np.mean(scores)
            print("{} for patient {}".format(score_means[p], p))
        return score_means, self.feature_columns

    def train(self):
        """
        build patient models
        """
        for p in tqdm(self.patients):
            features = self.train_data[p].drop([self.cls_col], axis=1)
            features = features[self.feature_columns]
            answers = self.train_data[p][self.cls_col]
            self.models[p].fit(features, answers)
            print(features.columns)
            print(self.models[p].feature_importances_)

    def get_info(self):
        return self.models

    def create_submission(self):
        import pandas as pd
        submission = []
        for p in self.all_patients:
            features = self.test_data[p].drop([self.mat_name_col], axis=1)
            features = features[self.feature_columns]
            mat_file_names = self.test_data[p][self.mat_name_col]
            if p in self.patients:
                answers = self.models[p].predict(features)
            else:
                answers = None
            for i in mat_file_names.index:
                submission.append({'File': mat_file_names[i], 'Class': 0 if answers is None else answers[i]})
        return pd.DataFrame(submission)

