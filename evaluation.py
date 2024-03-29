import matplotlib
from datetime import datetime
from seiz.segmented_feature_calc import SegmentedFeatureCalculator
from seiz.feature_io import FeatureIO
from seiz.learn import SegmentedOnePatientModelWrapper
from seiz.data_read import MatPatientDataReader
import os
import pandas as pd
import pickle
import gc

matplotlib.use("Agg")

models_dir = "../models"
fr = FeatureIO("../data/seg_features_5")


def get_model_path(patient, segment_size, tree_num, tag):
    path = os.path.join(models_dir, str(segment_size), str(tree_num))
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.join(path, "{}_{}.pcl".format(tag, patient))


def get_baseline_submission():
    submission = {}
    for p in [1, 2, 3]:
        dr = MatPatientDataReader(p, data_dir="../data")
        files = dr.get_test_files()
        for f in files:
            _, _, path = f
            mat_name = os.path.split(path)[1]
            submission[mat_name] = 0
    return submission


def train_and_save(patient, segment_size, tree_num, tag):
    model_path = get_model_path(patient, segment_size, tree_num, tag)
    print("Model will be saved at: " + model_path)
    model = SegmentedOnePatientModelWrapper(
        fr,
        patient_no=patient,
        segment_size=segment_size,
        number_of_trees=tree_num
    )
    model.train()
    rfc_model = model.get_info()
    with open(model_path, 'wb') as f:
        pickle.dump(rfc_model, f)


def evaluate_patient(patient, segment_size, tree_num, tag):
    model_path = get_model_path(patient, segment_size, tree_num, tag)
    print("Reading model {}...".format(model_path))
    with open(model_path, 'rb') as f:
        rfc_model = pickle.load(f)
    model = SegmentedOnePatientModelWrapper(
        fr,
        patient_no=patient,
        segment_size=segment_size,
        rfc_model=rfc_model
    )
    print("Starting evaluation...")
    return model.evaluate()


def calculate_raw_probs(segement_sizes, tree_nums, tag):
    submission = get_baseline_submission()
    print("Evaluating patient 1...")
    res_1 = evaluate_patient(1, segment_size=segement_sizes[0], tree_num=tree_nums[0], tag=tag)
    gc.collect()
    print("Evaluating patient 2...")
    res_2 = evaluate_patient(2, segment_size=segement_sizes[1], tree_num=tree_nums[1], tag=tag)
    gc.collect()
    print("Evaluating patient 3...")
    res_3 = evaluate_patient(3, segment_size=segement_sizes[2], tree_num=tree_nums[2], tag=tag)
    gc.collect()

    for x in res_1:
        submission[x['File']] = x['Class']
    for x in res_2:
        submission[x['File']] = x['Class']
    for x in res_3:
        submission[x['File']] = x['Class']

    fin_submission = []
    for k in submission.keys():
        fin_submission.append({'File': k, 'Class': submission[k]})

    df = pd.DataFrame(fin_submission)

    with open(os.path.join("probs", "probs_{}.pcl".format(tag)), 'wb') as f:
        pickle.dump(df, f)


def create_submissions(tag):
    with open(os.path.join("probs", "probs_{}.pcl".format(tag)), 'rb') as f:
        df = pickle.load(f)
        df.to_csv(os.path.join("submissions",
                               "submission_{}_{}.csv"
                               .format(tag, str(datetime.now()).replace(' ', '_'))),
                  index=False,
                  columns=["File", "Class"])


# calculate_raw_probs(segement_size=5, tree_num=700)
# create_submissions(segement_size=5, tree_num=700)

def main():
    tag = "5_2100_fet5"

    # train_and_save(patient=1, segment_size=5, tree_num=2100, tag=tag)
    # gc.collect()
    # train_and_save(patient=2, segment_size=5, tree_num=21, tag=tag)
    # gc.collect()
    # train_and_save(patient=3, segment_size=5, tree_num=1000, tag=tag)
    # gc.collect()

    calculate_raw_probs(segement_sizes=[5, 5, 5], tree_nums=[2100, 2100, 2100], tag=tag)
    create_submissions(tag=tag)


main()
