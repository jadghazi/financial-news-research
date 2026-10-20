from datetime import datetime, timezone
import unittest

from src.cmin_timing_study import eligible_time, selected_buckets


class TimingTests(unittest.TestCase):
    def test_delay_policies_are_explicit(self):
        self.assertEqual(eligible_time('2020-08-31', '', 'date0'), datetime(2020, 8, 31, tzinfo=timezone.utc))
        self.assertEqual(eligible_time('2020-08-31', '', 'delay1'), datetime(2020, 9, 1, tzinfo=timezone.utc))
        self.assertEqual(eligible_time('2020-08-31', '', 'delay2'), datetime(2020, 9, 2, tzinfo=timezone.utc))

    def test_clock_uses_dst_and_postclose_moves_forward(self):
        closes = [datetime(2020, 7, d, 20, tzinfo=timezone.utc) for d in (1, 2)]
        stories = [{'id': 'a', 'source_date': '2020-07-01', 'source_time_naive': '2020-07-01 15:59:00'},
                   {'id': 'b', 'source_date': '2020-07-01', 'source_time_naive': '2020-07-01 16:01:00'}]
        b = selected_buckets(stories, closes, 'clock_ny', 6)
        self.assertEqual([s['id'] for s in b[0]], ['a'])
        self.assertEqual([s['id'] for s in b[1]], ['b'])
        self.assertEqual(len(selected_buckets(stories, closes, 'date0', 6)[0]), 2)

    def test_exchange_early_close_and_weekend_mapping(self):
        closes = [datetime(2020, 11, 27, 18, tzinfo=timezone.utc), datetime(2020, 11, 30, 21, tzinfo=timezone.utc)]
        stories = [{'id': 'z', 'source_date': '2020-11-27', 'source_time_naive': '2020-11-27 13:01:00'},
                   {'id': 'a', 'source_date': '2020-11-28', 'source_time_naive': '2020-11-28 09:00:00'}]
        b = selected_buckets(stories, closes, 'clock_ny', 1)
        self.assertFalse(b[0])
        self.assertEqual([s['id'] for s in b[1]], ['a'])

    def test_unknown_policies_and_clock_interpretations_rejected(self):
        with self.assertRaises(ValueError): eligible_time('2020-08-31', '', 'invented')
        with self.assertRaises(ValueError): eligible_time('2020-08-31', '2020-08-31T12:00:00+00:00', 'clock_ny')


if __name__ == '__main__':
    unittest.main()
