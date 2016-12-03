import matplotlib

from seiz.data_read import MatPatientDataReader
from seiz.feature_io import FeatureIO
from seiz.segmented_feature_calc import SegmentedFeatureCalculator

# because of ssh
matplotlib.use("Agg")


def calc_one_patient(patient_no, train=True, test=True):
    feat_dir = "../data/seg_features_4"
    data_reader = MatPatientDataReader(patient_no=patient_no, data_dir="../data")
    fc = SegmentedFeatureCalculator(data_reader)
    fio = FeatureIO(feat_dir)
    if train:
        for seg_len in [1, 2, 4, 5, 6, 10]:
            print("Calculating train features, segment len = {}".format(seg_len))
            features_train = fc.calc_train_features(seg_len)
            print("Writing...")
            fio.write(features_train)
        print("OK")
    if test:
        for seg_len in [1, 2, 4, 5, 6, 10]:
            print("Calculating train features, segment len = {}".format(seg_len))
            features_test = fc.calc_test_features(seg_len)
            print("Writing...")
            fio.write(features_test)
        print("OK")


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
