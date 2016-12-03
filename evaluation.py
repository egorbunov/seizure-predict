import matplotlib
from datetime import datetime
from seiz.segmented_feature_calc import SegmentedFeatureCalculator
from seiz.feature_io import FeatureIO
from seiz.learn import SegmentedOnePatientModel

matplotlib.use("Agg")

# model = SeizureModel("../data/features3/train",
#                      "../data/features3/test",
#                      patients=[1, 2, 3],
#                      number_of_trees=100)
#
#
# def production(sub_name):
#     if not os.path.exists(os.path.split(sub_name)[0]):
#         os.makedirs(os.path.split(sub_name)[0])
#     model.train()
#     sub_df = model.create_submission()
#     print(model.get_info())
#     sub_df.to_csv(sub_name, columns=['File', 'Class'], index=False)
#
# model.do_bf_feature_selection()
# print("Final cross validation...")
# print(model.do_cross_validation())
#
# production('submissions/submission_{}.csv'
#            .format(str(datetime.now()).replace(' ', '_')))

fr = FeatureIO("../data/seg_features_1")
model = SegmentedOnePatientModel(fr, patient_no=1, segment_size=2, number_of_trees=100)
model.do_cross_validation()
model.train()
model.evaluate()

