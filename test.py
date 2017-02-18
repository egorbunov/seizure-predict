from seiz.feature_io import FeatureIO

fio = FeatureIO("../data/seg_features_4")
features = fio.read(patient=4, which="train", segment_size=2)
