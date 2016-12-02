import matplotlib
from seiz.feature_calculator import FeatureCalculator
import os
import pickle

# because of ssh
matplotlib.use("Agg")


def calc_one_patient(patient_no, train=True, test=True):
    features_train_dir = "../data/features3/train"
    features_test_dir = "../data/features3/test"
    if not os.path.exists(features_train_dir):
        os.makedirs(features_train_dir)
    if not os.path.exists(features_test_dir):
        os.makedirs(features_test_dir)
    fc = FeatureCalculator(patient_no, data_dir="../data")
    if train:
        print("Calculating train features...")
        features_train = fc.calc_train_features()
        features_train_file = os.path.join(features_train_dir, "{}.pcl".format(patient_no))
        with open(features_train_file, "wb") as f:
            pickle.dump(features_train, f)
    if test:
        print("Calculating test features...")
        features_test = fc.calc_test_features()
        features_test_file = os.path.join(features_test_dir, "{}.pcl".format(patient_no))
        with open(features_test_file, "wb") as f:
            pickle.dump(features_test, f)


def main(patient_no=None, train=True, test=True):
    if patient_no is not None:
        print("Calculating for one patient: {}...".format(patient_no))
        calc_one_patient(patient_no, train, test)
    else:
        print("Calculating for all patients!")
        for p in [1, 2, 3]:
            print("Patient: {}...".format(p))
            calc_one_patient(p)
            print("ok")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        main()
    else:
        train = False
        test = False
        for x in sys.argv:
            if x == "train":
                train = True
            elif x == "test":
                test = True
        main(int(sys.argv[1]), train, test)
