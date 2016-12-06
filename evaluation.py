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
fr = FeatureIO("../data/seg_features_4")


def get_model_path(patient, segment_size, tree_num):
    path = os.path.join(models_dir, str(segment_size), str(tree_num))
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.join(path, "{}.pcl".format(patient))


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


def train_and_save(patient, segment_size, tree_num):
    model_path = get_model_path(patient, segment_size, tree_num)
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


def train_and_save_all():
    train_and_save(patient=1, segment_size=2, tree_num=2000)
    gc.collect()
    train_and_save(patient=2, segment_size=2, tree_num=2000)
    gc.collect()
    train_and_save(patient=3, segment_size=2, tree_num=2000)
    gc.collect()


def evaluate_patient(patient, segment_size, tree_num):
    model_path = get_model_path(patient, segment_size, tree_num)
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


def calculate_raw_probs(segement_size, tree_num):
    submission = get_baseline_submission()
    print("Evaluating patient 1...")
    res_1 = evaluate_patient(1, segment_size=2, tree_num=tree_num)
    gc.collect()
    print("Evaluating patient 2...")
    res_2 = evaluate_patient(2, segment_size=segement_size, tree_num=tree_num)
    gc.collect()
    print("Evaluating patient 3...")
    res_3 = evaluate_patient(3, segment_size=segement_size, tree_num=tree_num)
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

    with open(os.path.join("probs", "probs_{}_{}".format(segement_size, tree_num)), 'wb') as f:
        pickle.dump(df, f)


def create_submissions(segement_size=5, tree_num=2000):
    with open(os.path.join("probs", "probs_{}_{}".format(segement_size, tree_num)), 'rb') as f:
        df = pickle.load(f)
        df.to_csv(os.path.join("submissions",
                               "submission_{}_{}_{}.csv"
                               .format(segement_size, tree_num, str(datetime.now()).replace(' ', '_'))),
                  index=False,
                  columns=["File", "Class"])


# calculate_raw_probs(segement_size=5, tree_num=700)
# create_submissions(segement_size=5, tree_num=700)

# train_and_save(patient=1, segment_size=2, tree_num=700)
# train_and_save(patient=2, segment_size=5, tree_num=700)
# train_and_save(patient=3, segment_size=5, tree_num=700)
