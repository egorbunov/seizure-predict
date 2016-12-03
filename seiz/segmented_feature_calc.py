import numpy as np
import pandas as pd
import scipy.signal
import sklearn.preprocessing
from tqdm import tqdm
import gc
# from pympler import muppy
# from pympler import summary


class SegmentedFeatureCalculator:
    def __init__(self, patient_data_reader):
        """
        :param patient_data_reader: MatPatientDataReader
        """
        self.samples_reader = patient_data_reader

    @staticmethod
    def _calc_small_sample(spat_sample, fft_sample):
        """
        takes sample in spatial domain ((NxM), where N -- number of channels)
        and same sample in frequency domain
        """
        # spatial features
        scaled = sklearn.preprocessing.scale(spat_sample, axis=0)
        corr_mat = np.corrcoef(scaled)
        feature_dict = {}
        for i in range(corr_mat.shape[0]):
            for j in range(i + 1, corr_mat.shape[1]):
                feature_dict['spatial_corr_{}_{}'.format(i, j)] = corr_mat[i, j]
        try:
            eigenvalues = np.absolute(np.linalg.eig(corr_mat)[0])
        except np.linalg.LinAlgError as e:
            return None

        for i in range(len(eigenvalues)):
            feature_dict['sp_corr_eig_{}'.format(i)] = eigenvalues[i]
        # frequency features
        abs_fft = np.absolute(fft_sample)
        scaled_freq = sklearn.preprocessing.scale(abs_fft, axis=0)
        corr_freq_mat = np.corrcoef(scaled_freq)
        for i in range(corr_freq_mat.shape[0]):
            for j in range(i + 1, corr_freq_mat.shape[1]):
                feature_dict['freq_corr_{}_{}'.format(i, j)] = corr_freq_mat[i, j]
        try:
            eigenvalues_freq = np.absolute(np.linalg.eig(corr_freq_mat)[0])
        except np.linalg.LinAlgError as e:
            return None
        for i in range(len(eigenvalues_freq)):
            feature_dict['freq_corr_eig_{}'.format(i)] = eigenvalues_freq[i]

        return feature_dict

    @staticmethod
    def _features_from_one_sample(big_sample, additional_info, segment_len):
        n_first_freq = 50
        signals = big_sample['signals']
        rate = big_sample['sampling_rate']
        # list of shape (split num, electrodes num, segment size)
        time_split = np.split(signals.as_matrix().T, signals.shape[0] / (rate * segment_len), axis=1)
        resampled_time_split = scipy.signal.resample(time_split, num=400, axis=2)
        # for frequency domain features
        fft_split = np.fft.rfft(time_split, axis=2)[:, :, :n_first_freq]
        for spat, freq in zip(resampled_time_split, fft_split):
            features = SegmentedFeatureCalculator._calc_small_sample(spat, freq)
            if features is None:
                continue
            yield {**additional_info, **features}

        # all_objects = muppy.get_objects()
        # sumr = summary.summarize(all_objects)
        # summary.print_(sumr)

    @staticmethod
    def _calc_for_one_train_sample(big_sample, segment_len):
        return SegmentedFeatureCalculator._features_from_one_sample(
            big_sample, {'#cls': big_sample['cls']}, segment_len
        )

    @staticmethod
    def _calc_for_one_test_sample(big_sample, segment_len):
        return SegmentedFeatureCalculator._features_from_one_sample(
            big_sample, {'#mat_name': big_sample['mat_name']}, segment_len
        )

    def calc_train_features(self, segment_len):
        big_samples, cnt = self.samples_reader.train_samples()

        def gen():
            with tqdm(range(cnt)) as timer:
                for bs in big_samples:
                    yield from SegmentedFeatureCalculator._calc_for_one_train_sample(bs, segment_len)
                    gc.collect()
                    timer.update()

        return {'features': pd.DataFrame(gen()),
                'segment_size': segment_len,
                'patient': self.samples_reader.patient_no,
                'which': "train"}

    def calc_test_features(self, segment_len):
        big_samples, cnt = self.samples_reader.test_samples()

        def gen():
            with tqdm(range(cnt)) as timer:
                for bs in big_samples:
                    yield from SegmentedFeatureCalculator._calc_for_one_test_sample(bs, segment_len)
                    gc.collect()
                    timer.update()

        return {'features': pd.DataFrame(gen()),
                'segment_size': segment_len,
                'patient': self.samples_reader.patient_no,
                'which': "test"}
