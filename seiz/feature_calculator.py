import numpy as np
from tqdm import tqdm
from seiz.data_read import MatPatientDataReader


class FeatureCalculator:
    def __init__(self, patient_no, data_dir):
        self.data_reader = MatPatientDataReader(patient_no, data_dir)

    @staticmethod
    def calc_features_one_sample(df):
        """
        :param df: dataframe with signals from electrodes
        :return:
        """
        #smmoting
        df = df.rolling(window=22).mean().dropna()

        sum_signal = df.sum(axis=1).as_matrix()
        sum_std = np.std(sum_signal)
        sum_mean = np.mean(sum_signal)
        sum_perc_30 = np.percentile(sum_signal, 30)
        sum_perc_70 = np.percentile(sum_signal, 70)

        sum_fft = np.abs(np.fft.fft(sum_signal))
        sum_fft_std = np.std(sum_fft)
        sum_fft_mean = np.mean(sum_fft)
        sum_fft_perc_30 = np.percentile(sum_fft, 30)
        sum_fft_perc_70 = np.percentile(sum_fft, 70)

        std = np.std(df.as_matrix(), axis=0)
        mean = np.mean(df.as_matrix(), axis=0)
        perc_30 = np.percentile(df.as_matrix(), 30, axis=0)
        perc_70 = np.percentile(df.as_matrix(), 70, axis=0)

        fft = np.abs(np.fft.fft(df.as_matrix(), axis=0))

        fft_std = np.std(fft, axis=0)
        fft_mean = np.mean(fft, axis=0)
        fft_perc_30 = np.percentile(fft, 30, axis=0)
        fft_perc_70 = np.percentile(fft, 70, axis=0)

        features_dict = {
            'sum_std': sum_std,
            'sum_mean': sum_mean,
            'sum_perc_30': sum_perc_30,
            'sum_perc_70': sum_perc_70,
            'sum_fft_std': sum_fft_std,
            'sum_fft_mean': sum_fft_mean,
            'sum_fft_perc_30': sum_fft_perc_30,
            'sum_fft_perc_70': sum_fft_perc_70
        }

        for i in range(len(df.columns)):
            features_dict["{}_std".format(i)] = std[i]
            features_dict["{}_fft_std".format(i)] = fft_std[i]
            features_dict["{}_mean".format(i)] = mean[i]
            features_dict["{}_fft_mean".format(i)] = fft_mean[i]
            features_dict["{}_perc_30".format(i)] = perc_30[i]
            features_dict["{}_fft_perc_30".format(i)] = fft_perc_30[i]
            features_dict["{}_perc_70".format(i)] = perc_70[i]
            features_dict["{}_fft_perc_70".format(i)] = fft_perc_70[i]

        return features_dict

    @staticmethod
    def _calc_features_train_wrapper(train_sample):
        features = FeatureCalculator.calc_features_one_sample(train_sample['df'])
        return {'#cls': train_sample['cls'], **features}

    @staticmethod
    def _calc_features_test_wrapper(test_sample):
        features = FeatureCalculator.calc_features_one_sample(test_sample['df'])
        return {'#mat_name': test_sample['mat_name'], **features}

    def calc_train_features(self):
        """
        Calculates features for train samples
        :return:
        """
        train_samples, train_sz = self.data_reader.get_train_data_generator()
        all_features = []
        with tqdm(range(train_sz)) as t:
            for ts in train_samples:
                features = FeatureCalculator._calc_features_train_wrapper(ts)
                all_features.append(features)
                t.update()
        return all_features

    def calc_test_features(self):
        """
        Calculates features for test samples
        :return:
        """
        test_samples, test_sz = self.data_reader.get_test_data_generator()
        all_features = []
        with tqdm(range(test_sz)) as t:
            for ts in test_samples:
                features = FeatureCalculator._calc_features_test_wrapper(ts)
                all_features.append(features)
                t.update()
        return all_features