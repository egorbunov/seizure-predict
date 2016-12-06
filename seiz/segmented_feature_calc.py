import numpy as np
import scipy.signal
import sklearn.preprocessing
from tqdm import tqdm
from scipy.stats import spearmanr


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
        def get_features(sample, prefix):
            scaled = sklearn.preprocessing.scale(sample, axis=0)
            corr_mat = spearmanr(scaled, axis=1)[0]
            feature_dict = {}
            for i in range(corr_mat.shape[0]):
                for j in range(i + 1, corr_mat.shape[1]):
                    feature_dict['{}_spatial_corr_{}_{}'.format(prefix, i, j)] = corr_mat[i, j]
            try:
                eigenvalues = np.absolute(np.linalg.eig(corr_mat)[0])
                eigenvalues.sort()
                for i in range(len(eigenvalues)):
                    feature_dict['{}_corr_eig_{}'.format(prefix, i)] = eigenvalues[i]
            except np.linalg.LinAlgError as e:
                return None
            return feature_dict

        # spatial features
        spatial_features = get_features(spat_sample, "spatial")
        # frequency features
        abs_fft = np.absolute(fft_sample)
        freq_features = get_features(abs_fft, "freq")

        if spatial_features is None or freq_features is None:
            return None

        return {**spatial_features, **freq_features}

    @staticmethod
    def _features_from_one_sample(big_sample, additional_info, segment_len):
        n_first_freq = 47
        signals = big_sample['signals']
        rate = big_sample['sampling_rate']
        # list of shape (split num, electrodes num, segment_len * rate)
        time_split = np.split(signals.T, signals.shape[0] / (rate * segment_len), axis=1)
        # resampled_time_split = scipy.signal.resample(time_split,
        #                                              num=min(400, np.shape(time_split)[2]),
        #                                              axis=2)
        # for frequency domain features
        fft_split = np.fft.rfft(time_split, axis=2)[:, :, :n_first_freq]
        for spat, freq in zip(time_split, fft_split):
            features = SegmentedFeatureCalculator._calc_small_sample(spat, freq)
            if features is None:
                continue
            yield {**additional_info, **features}

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
                    timer.update()

        return {'features': gen(),
                'segment_size': segment_len,
                'patient': self.samples_reader.patient_no,
                'which': "train"}

    def calc_test_features(self, segment_len):
        big_samples, cnt = self.samples_reader.test_samples()

        def gen():
            with tqdm(range(cnt)) as timer:
                for bs in big_samples:
                    yield from SegmentedFeatureCalculator._calc_for_one_test_sample(bs, segment_len)
                    timer.update()

        return {'features': gen(),
                'segment_size': segment_len,
                'patient': self.samples_reader.patient_no,
                'which': "test"}
