"""Startup marker removal must preserve the shared time base and DI3 ordinals."""
import unittest
import warnings
from unittest.mock import patch

import numpy as np

import harp_utils
from harp_utils import qc_continuous_harp
from slap2_continuous_sync import build_continuous_clock, map_continuous_lines


class ContinuousHarpStartTests(unittest.TestCase):
    def test_short_startup_pair_before_unclosed_acquisition(self):
        data = self.fixture()
        for key in ('slap2_end_times', 'normalized_slap2_end', 'slap2_end_signal'):
            data[key] = data[key][:1]
        # Legacy cleanup still requires another complete pair.
        legacy = harp_utils.trim_leading_trial_pulse_artifact(data)
        np.testing.assert_array_equal(legacy['normalized_slap2_start'], [0, 4])
        with self.assertWarnsRegex(RuntimeWarning, 'Removed erroneous leading'):
            trimmed = harp_utils.trim_leading_trial_pulse_artifact(data, continuous=True)
        qc_continuous_harp(trimmed)
        for key in ('slap2_start_times', 'normalized_slap2_start'):
            np.testing.assert_array_equal(trimmed[key], [4])
        for key in ('slap2_end_times', 'normalized_slap2_end', 'slap2_end_signal'):
            self.assertEqual(trimmed[key].size, 0)
        np.testing.assert_array_equal(trimmed['slap2_start_signal'], [100])
        np.testing.assert_array_equal(trimmed['grating_times'], [5])
        np.testing.assert_array_equal(trimmed['normalized_start_gratings'], [5])
        np.testing.assert_array_equal(trimmed['grating_signal'], [100])
        for key in ('time_reference', 'recording_start_time', 'analog_times',
                    'normalized_analog_times', 'slap2_cycle_clock_times',
                    'normalized_slap2_cycle_clock_times', 'slap2_cycle_clock_signal',
                    'photodiode', 'wheel'):
            self.assertIs(trimmed[key], data[key])
        np.testing.assert_array_equal(data['normalized_slap2_start'], [0, 4])
        np.testing.assert_array_equal(data['normalized_slap2_end'], [0.002])

    def test_continuous_short_pair_needs_no_later_pair_or_duration_comparison(self):
        for starts, ends in (([0], [0.002]), ([0, 4], [0.002, 4.003])):
            with self.subTest(starts=starts, ends=ends):
                data = self.fixture()
                for key in ('slap2_start_times', 'normalized_slap2_start'):
                    data[key] = np.array(starts, dtype=float)
                for key in ('slap2_end_times', 'normalized_slap2_end'):
                    data[key] = np.array(ends, dtype=float)
                data['slap2_start_signal'] = np.ones(len(starts))
                data['slap2_end_signal'] = np.ones(len(ends))
                with self.assertWarns(RuntimeWarning):
                    trimmed = harp_utils.trim_leading_trial_pulse_artifact(data, continuous=True)
                np.testing.assert_array_equal(trimmed['normalized_slap2_start'], starts[1:])
                np.testing.assert_array_equal(trimmed['normalized_slap2_end'], ends[1:])
                if len(starts) == 1:
                    self.assertEqual(trimmed['grating_times'].size, 0)
                    with self.assertRaisesRegex(ValueError, 'exactly one retained'):
                        qc_continuous_harp(trimmed)

    def test_continuous_cleanup_preserves_missing_or_nonshort_pairs(self):
        for ends in ([], [0], [-0.002], [0.1], [1], [np.nan], [np.inf]):
            with self.subTest(ends=ends):
                data = self.fixture()
                data['slap2_end_times'] = data['normalized_slap2_end'] = np.array(ends)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    trimmed = harp_utils.trim_leading_trial_pulse_artifact(data, continuous=True)
                self.assertEqual(len(caught), 0)
                for key in data:
                    self.assertIs(trimmed[key], data[key])

    def test_marker_validation_does_not_repeat_shared_trimming(self):
        data = self.fixture()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            trimmed = harp_utils.trim_leading_trial_pulse_artifact(data)
        original = dict(trimmed)
        with patch.object(harp_utils, 'trim_leading_trial_pulse_artifact') as trim:
            actual = qc_continuous_harp(trimmed)
        trim.assert_not_called()
        self.assertIsNone(actual)
        self.assertEqual(trimmed.keys(), original.keys())
        for key in original:
            self.assertIs(trimmed[key], original[key])

    def fixture(self):
        data = {'time_reference': 100.0, 'recording_start_time': -0.04,
                'slap2_start_signal': np.array([100, 100]),
                'slap2_end_signal': np.array([100, 100]),
                'slap2_cycle_clock_signal': np.array([1, 0, 1, 0, 1]),
                'photodiode': np.array([0, 1, 0]), 'wheel': np.array([2, 3, 4]),
                'grating_signal': np.array([100, 100])}
        for key, alias, values in (
            ('slap2_start_times', 'normalized_slap2_start', [0, 4]),
            ('slap2_end_times', 'normalized_slap2_end', [0.002, 10]),
            ('slap2_cycle_clock_times', 'normalized_slap2_cycle_clock_times', [4.001, 4.01, 5.001, 5.01, 6.001]),
            ('analog_times', 'normalized_analog_times', [-0.04, 4.001, 10]),
            ('grating_times', 'normalized_start_gratings', [0.003, 5]),
        ):
            data[key] = data[alias] = np.array(values, dtype=float)
        return data

    def test_startup_removal_preserves_original_origin_and_retained_timestamps(self):
        data = self.fixture()
        originals = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}
        with self.assertWarnsRegex(RuntimeWarning, 'Removed erroneous leading'):
            trimmed = harp_utils.trim_leading_trial_pulse_artifact(data)
        qc_continuous_harp(trimmed)
        result = trimmed
        self.assertEqual(result['time_reference'], 100)
        self.assertAlmostEqual(result['recording_start_time'], -0.04)
        np.testing.assert_array_equal(result['normalized_slap2_start'], [4])
        np.testing.assert_array_equal(result['normalized_slap2_end'], [10])
        for key in ('slap2_cycle_clock_times', 'analog_times'):
            self.assertIs(result[key], data[key])
            np.testing.assert_allclose(result[key] + result['time_reference'], data[key] + data['time_reference'])
        # Shared startup cleanup removes early DO2 events; retained events still
        # use the original clock. Movie synchronization does not use DO2.
        np.testing.assert_array_equal(result['grating_times'], [5])
        np.testing.assert_allclose(result['grating_times'] + result['time_reference'],
                                   data['grating_times'][1:] + data['time_reference'])
        np.testing.assert_array_equal(result['grating_signal'], data['grating_signal'][1:])
        self.assertIs(result['grating_times'], result['normalized_start_gratings'])
        for key in ('slap2_cycle_clock_signal', 'photodiode', 'wheel'):
            np.testing.assert_array_equal(result[key], data[key])
        self.assertIs(result['analog_times'], result['normalized_analog_times'])
        self.assertIs(result['slap2_cycle_clock_times'], result['normalized_slap2_cycle_clock_times'])
        for key in data:
            np.testing.assert_array_equal(data[key], originals[key])
        self.assertEqual(result.keys(), data.keys())

    def test_sample_interpolation_is_unchanged_without_dropping_pulses(self):
        data = self.fixture()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            trimmed = harp_utils.trim_leading_trial_pulse_artifact(data)
        qc_continuous_harp(trimmed)
        clocks = [build_continuous_clock(d['slap2_cycle_clock_signal'], d['slap2_cycle_clock_times'], 10, 2,
                         recording_start=d['recording_start_time']) for d in (data, trimmed)]
        self.assertEqual(clocks[1]['qc']['detected_pulse_count'], 3)
        np.testing.assert_allclose(clocks[1]['cycle_starts'], [4.001, 5.001, 6.001])
        self.assertFalse(clocks[1]['qc']['onset_warning'])
        np.testing.assert_allclose(map_continuous_lines([1, 5, 11, 20], clocks[1]),
                                   map_continuous_lines([1, 5, 11, 20], clocks[0]))

    def test_single_real_start_without_end_is_preserved(self):
        data = self.fixture()
        for key in ('normalized_slap2_start', 'slap2_start_times'):
            data[key] = np.array([0.0])
        for key in ('normalized_slap2_end', 'slap2_end_times'):
            data[key] = np.array([])
        data['slap2_start_signal'] = np.array([100])
        data['slap2_end_signal'] = np.array([])
        trimmed = harp_utils.trim_leading_trial_pulse_artifact(data)
        original = dict(trimmed)
        self.assertIsNone(qc_continuous_harp(trimmed))
        self.assertEqual(trimmed.keys(), original.keys())
        for key in original:
            self.assertIs(trimmed[key], original[key])
        np.testing.assert_array_equal(trimmed['normalized_slap2_start'], [0.0])
        self.assertEqual(trimmed['normalized_slap2_end'].size, 0)

    def test_multiple_retained_pairs_are_rejected_not_collapsed(self):
        data = self.fixture()
        data['slap2_end_times'][0] = 1.0
        trimmed = harp_utils.trim_leading_trial_pulse_artifact(data)
        with self.assertRaisesRegex(ValueError, 'exactly one retained'):
            qc_continuous_harp(trimmed)
        np.testing.assert_array_equal(trimmed['normalized_slap2_start'], [0, 4])
        np.testing.assert_array_equal(trimmed['normalized_slap2_end'], [1, 10])
        self.assertEqual(trimmed['time_reference'], 100)

    def test_marker_validation_rejects_missing_or_extra_markers(self):
        for starts, ends in (([], [10]), ([4, 5], []), ([4, 5], [10]),
                             ([4], [9, 10]), ([], [])):
            with self.subTest(starts=starts, ends=ends):
                data = self.fixture()
                data['normalized_slap2_start'] = np.asarray(starts)
                data['normalized_slap2_end'] = np.asarray(ends)
                with self.assertRaisesRegex(ValueError, 'exactly one retained'):
                    qc_continuous_harp(data)

    def test_marker_validation_rejects_nonfinite_or_reversed_pair(self):
        for start, stop, message in ((np.nan, 10, 'finite'), (4, np.inf, 'finite'),
                                     (4, np.nan, 'finite'), (4, 4, 'after'),
                                     (4, 3, 'after')):
            with self.subTest(start=start, stop=stop):
                data = self.fixture()
                data['normalized_slap2_start'] = np.asarray([start])
                data['normalized_slap2_end'] = np.asarray([stop])
                with self.assertRaisesRegex(ValueError, message):
                    qc_continuous_harp(data)

    def test_marker_validation_rejects_inconsistent_aliases(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            data = harp_utils.trim_leading_trial_pulse_artifact(self.fixture())
        data['normalized_analog_times'] = data['analog_times'] + 1
        with self.assertRaisesRegex(ValueError, 'Inconsistent HARP timestamp aliases'):
            qc_continuous_harp(data)

    def test_legacy_trim_does_not_rebase_and_still_filters_do2(self):
        data = self.fixture()
        with self.assertWarns(RuntimeWarning):
            result = harp_utils.trim_leading_trial_pulse_artifact(data)
        np.testing.assert_array_equal(result['normalized_slap2_start'], [4])
        np.testing.assert_array_equal(result['grating_times'], [5])
        self.assertEqual(result['time_reference'], 100)
        self.assertIs(result['slap2_cycle_clock_times'], data['slap2_cycle_clock_times'])


if __name__ == '__main__':
    unittest.main()