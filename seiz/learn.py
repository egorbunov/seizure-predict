from seiz.feature_read import FeatureReader
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm


class SeizureModel:
    def __init__(self, feature_train_dir, feature_test_dir, patients, number_of_trees=2000):
        self.feature_reader = FeatureReader(feature_train_dir, feature_test_dir)
        self.patients = patients
        self.cls_col = '#cls'
        self.mat_name_col = '#mat_name'
        self.models = {}
        self.train_data = {}
        self.test_data = {}
        self.all_patients = [1, 2, 3]
        self.feature_names = {}
        self.all_feature_names = {}

        # creating models...
        for p in patients:
            if p not in set(self.all_patients):
                raise RuntimeError("Bad patient id")
            self.models[p] = RandomForestClassifier(
                n_estimators=number_of_trees,
                n_jobs=8
            )
            self.train_data[p] = self.feature_reader.read_train_dataset([p])
            self.all_feature_names[p] = self.train_data[p].drop(self.cls_col, axis=1).columns
            self.feature_names[p] = self.train_data[p].drop(self.cls_col, axis=1).columns

    def do_importance_feature_selection(self):
        for p in self.patients:
            print("Patient: {}".format(p))
            self.feature_names[p] = self.all_feature_names[p]
            max_score = 0
            while True:
                print("Feature cnt: {}".format(len(self.feature_names[p])))
                score, importance = self.do_cross_validation_patient(p)
                if score > max_score:
                    max_score = score
                    print("Gor score: {}".format(score))
                    min_idx = importance.argmin()
                    self.feature_names[p] = np.delete(self.feature_names[p], min_idx)
                else:
                    break
        return self.feature_names

    def do_bf_feature_selection(self):
        for p in self.patients:
            print("Processing patient {}".format(p))
            f_num = len(self.feature_names[p])
            n = 10
            stop_f_num = 3
            max_score = 0
            best_params = []
            while f_num > stop_f_num:
                made_better = False
                print("feature num = {}; choices cnt = {}".format(f_num, n))
                print("Performing cross validation on randomly chosen features {} times".format(n))
                for _ in tqdm(range(n)):
                    self.feature_names[p] = np.random.choice(self.all_feature_names[p], f_num, replace=False)
                    score, _ = self.do_cross_validation_patient(p, 3)
                    if score > max_score:
                        made_better = True
                        print("Got score: {}".format(score))
                        max_score = score
                        best_params = list(self.feature_names[p])
                if not made_better:
                    break
                n = int(np.ceil(n * 1.1))
                f_num = int(f_num * 0.7)
            self.feature_names[p] = best_params
        return self.feature_names

    def do_cross_validation_patient(self, patient, n_splits=10):
        skf = StratifiedKFold(n_splits=n_splits)
        features = self.train_data[patient][self.feature_names[patient]].as_matrix()
        answers = self.train_data[patient][self.cls_col].as_matrix()
        scores = []
        imps = np.array([[0]*len(self.feature_names[patient])])
        for train_index, test_index in skf.split(features, answers):
            f_train, f_test = features[train_index], features[test_index]
            a_train, a_test = answers[train_index], answers[test_index]
            self.models[patient].fit(f_train, a_train)
            scores.append(self.models[patient].score(f_test, a_test))
            imps = np.concatenate((imps, np.array([self.models[patient].feature_importances_])))
        return np.mean(scores), np.mean(imps, axis=0)

    def do_cross_validation(self, n_splits=10):
        score_means = {}
        for p in tqdm(self.patients):
            score_means[p], _ = self.do_cross_validation_patient(p, n_splits)
        return score_means

    def train(self):
        """
        build patient models
        """
        for p in tqdm(self.patients):
            features = self.train_data[p][self.feature_names[p]].as_matrix()
            answers = self.train_data[p][self.cls_col]
            self.models[p].fit(features, answers)

    def get_info(self):
        return self.models

    def create_submission(self):
        # reading test features
        for p in self.all_patients:
            self.test_data[p] = self.feature_reader.read_test_dataset([p])

        import pandas as pd
        submission = []
        for p in self.all_patients:
            if p in self.patients:
                features = self.test_data[p][self.feature_names[p]]
                mat_file_names = self.test_data[p][self.mat_name_col]
                answers = self.models[p].predict(features)
            else:
                answers = None
            for i in mat_file_names.index:
                submission.append({'File': mat_file_names[i], 'Class': 0 if answers is None else answers[i]})
        return pd.DataFrame(submission)

