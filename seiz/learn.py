from collections import Counter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import pandas as pd


class SegmentedOnePatientModelWrapper:
    def __init__(self, feature_io, patient_no, number_of_trees=100, segment_size=None, rfc_model=None):
        self.patient_no = patient_no
        self.segment_size = segment_size
        self.feature_io = feature_io
        self.cls_col = '#cls'
        self.mat_name_col = '#mat_name'
        if rfc_model is None:
            self.model = RandomForestClassifier(
                n_estimators=number_of_trees,
                n_jobs=17
            )
        else:
            self.model = rfc_model
        self.train_data = None
        self.test_data = None
        self.all_feature_names = None
        self.feature_names = None

    def _read_train(self):
        if self.train_data is not None:
            return
        train_features_lst = self.feature_io.read(patient=self.patient_no,
                                                  which="train",
                                                  segment_size=self.segment_size)
        self.train_data = pd.DataFrame(train_features_lst)
        self.train_data[self.cls_col] = self.train_data[self.cls_col].astype('category')
        self.all_feature_names = self.train_data.drop(self.cls_col, axis=1).columns
        self.feature_names = self.train_data.drop(self.cls_col, axis=1).columns

    def _read_test(self):
        if self.test_data is not None:
            return
        test_features_lst = self.feature_io.read(patient=self.patient_no,
                                                 which="test",
                                                 segment_size=self.segment_size)
        self.test_data = pd.DataFrame(test_features_lst)
        self.all_feature_names = self.test_data.drop(self.mat_name_col, axis=1).columns
        self.feature_names = self.test_data.drop(self.mat_name_col, axis=1).columns

    def do_bf_feature_selection(self):
        self._read_train()
        print("Processing patient {}".format(self.patient_no))
        f_num = 5
        n = 100
        stop_f_num = len(self.feature_names)
        max_score = 0
        best_params = []
        while f_num < stop_f_num:
            print("feature num = {}; choices cnt = {}".format(f_num, n))
            print("Performing cross validation on randomly chosen features {} times".format(n))
            for _ in tqdm(range(n)):
                self.feature_names = np.random.choice(self.all_feature_names, f_num, replace=False)
                score, _ = self.do_cross_validation(n_splits=3)
                if score > max_score:
                    print("Got score: {}".format(score))
                    max_score = score
                    best_params = list(self.feature_names)
            # n = int(np.ceil(n / 1.1))
            f_num = int(f_num * 1.5)
        self.feature_names = best_params

    def do_cross_validation(self, n_splits=10):
        self._read_train()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        features = self.train_data[self.feature_names].as_matrix()
        answers = self.train_data[self.cls_col].as_matrix()
        scores = []
        imps = np.array([[0] * len(self.feature_names)])
        with tqdm(range(n_splits)) as timer:
            for train_index, test_index in skf.split(features, answers):
                f_train, f_test = features[train_index], features[test_index]
                a_train, a_test = answers[train_index], answers[test_index]
                self.model.verbose = 0
                self.model.fit(f_train, a_train)
                score = self.model.score(f_test, a_test)
                scores.append(score)
                print(score)
                imps = np.concatenate((imps, np.array([self.model.feature_importances_])))
                timer.update()
        return np.mean(scores), np.mean(imps, axis=0)

    def train(self):
        self._read_train()
        """
        build patient models
        """
        print("Features: {}".format(self.feature_names))
        features = self.train_data[self.feature_names].as_matrix()
        print("Train segments shape: {}".format(features.shape))
        answers = self.train_data[self.cls_col]
        self.model.verbose = 2
        self.model.fit(features, answers)

    def get_info(self):
        return self.model

    def evaluate(self):
        self._read_test()
        one_class_idx = 0 if self.model.classes_[0] == 1 else 1
        # reading test features
        mat_names = self.test_data['#mat_name'].unique()
        result = []
        for mat_name in tqdm(mat_names):
            one_sample_segments = self.test_data.loc[self.test_data[self.mat_name_col] == mat_name]
            features = one_sample_segments[self.feature_names]
            self.model.verbose = 0
            classes_probs = self.model.predict_proba(features)
            one_class_probs = classes_probs[:, one_class_idx]
            sum_prob = np.median(one_class_probs)
            ans = {'File': mat_name, 'Class': sum_prob}
            print(ans)
            result.append(ans)
        return result
