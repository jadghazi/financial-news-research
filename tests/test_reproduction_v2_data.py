"""Synthetic-file tests only; real development data are not prepared here."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from src.reproduction_v2_data import TICKER_DESCRIPTIONS, parse_created_at, prepare, previous_session


class ReproductionV2DataTests(unittest.TestCase):
    def fixture(self, root, messages=None, future_bad=True):
        source = root / "source"
        (source / "price/raw").mkdir(parents=True)
        dates = ["2013-12-31", "2014-01-02", "2014-01-03", "2014-01-06", "2014-01-07",
                 "2015-06-30", "2015-07-01", "2015-09-30", "2015-10-01", "2015-10-02"]
        for ticker in TICKER_DESCRIPTIONS:
            with (source / f"price/raw/{ticker}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
                for i, date in enumerate(dates):
                    if future_bad and date >= "2015-10-01":
                        writer.writerow([date, *("UNSEEN-INVALID" for _ in range(6))])
                    elif date == "2014-01-06":
                        # Split-like adjustment: raw close falls while adjusted rises.
                        writer.writerow([date, 52, 54, 50, 51, 102.25, 1000])
                    else:
                        writer.writerow([date, 100+i, 102+i, 99+i, 100+i, 100+i, 1000+i])
            folder = source / f"tweet/preprocessed/{ticker}"
            folder.mkdir(parents=True)
            with (folder / "2014-01-03").open("w", encoding="utf-8") as stream:
                for row in messages or []:
                    stream.write(json.dumps(row) + "\n")
        return source

    def message(self, time, text):
        return {"created_at": time, "text": text.split()}

    def test_actual_utc_cutoffs_weekend_and_filename_mismatch(self):
        messages = [
            self.message("Fri Jan 03 22:00:00 +0000 2014", "excluded boundary"),
            self.message("Sat Jan 04 10:00:00 +0000 2014", "weekend news"),
            self.message("Mon Jan 06 12:00:00 +0000 2014", "included exact preopen"),
            self.message("Mon Jan 06 12:00:01 +0000 2014", "post preopen"),
            self.message("Mon Jan 06 22:00:00 +0000 2014", "included exact postclose"),
            self.message("Mon Jan 06 22:00:01 +0000 2014", "excluded postclose"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root, messages)
            records, manifest = prepare(source, root / "output")
            pre = next(r for r in records["pre_open"] if r["ticker"] == "AAPL" and r["target_date"] == "2014-01-06")
            self.assertEqual(pre["cutoff_utc"], "2014-01-06T12:00:00Z")
            self.assertEqual(pre["forecast_date"], "2014-01-06")
            self.assertEqual(pre["price_date"], "2014-01-03")
            self.assertEqual(pre["target_available_utc"], "2014-01-06T22:00:00Z")
            self.assertEqual({r["text"] for r in pre["headlines"]}, {"weekend news", "included exact preopen"})
            post = next(r for r in records["post_close"] if r["ticker"] == "AAPL" and r["forecast_date"] == "2014-01-06")
            self.assertEqual({r["text"] for r in post["headlines"]}, {"weekend news", "included exact preopen", "post preopen", "included exact postclose"})
            self.assertEqual(len(manifest["message_audits"]["AAPL"]["filename_timestamp_mismatches"]), 5)

    def test_closed_base_prices_adjusted_labels_and_unseen_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records, manifest = prepare(self.fixture(root), root / "output")
            rows = records["pre_open"]
            row = next(r for r in rows if r["ticker"] == "AAPL" and r["target_date"] == "2014-01-06")
            self.assertEqual(row["price_date"], "2014-01-03")
            self.assertEqual(row["forecast_date"], "2014-01-06")
            self.assertEqual(row["market_raw"][3:5], [102., 102.])
            self.assertEqual(row["target_up"], 1)  # adjusted increase despite raw split decline
            self.assertAlmostEqual(row["target_return"], 102.25 / 102 - 1)
            self.assertEqual(row["open_up"], 1)
            self.assertFalse(row["band_eligible"])
            self.assertEqual(row["market_relative"][3], 102 / 101 - 1)
            self.assertFalse(any(r["target_date"] >= "2015-10-01" for r in rows))
            self.assertNotIn("test", manifest["summary"]["pre_open"])
            self.assertNotIn("test", manifest["split_ranges"])
            self.assertIn("2015-10-01", manifest["common_calendar"])
            self.assertIn("price/raw/AAPL.csv", manifest["source_sha256"])
            self.assertEqual(next(r for r in rows if r["target_date"] == "2015-07-01")["split"], "valid")

    def test_empty_dates_dedup_earliest_and_hash_cap_do_not_use_labels(self):
        messages = [self.message("Thu Jan 02 15:00:00 +0000 2014", f"headline {i}") for i in range(12)]
        messages += [self.message("Thu Jan 02 14:00:00 +0000 2014", "headline 0")]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records, _ = prepare(self.fixture(root, messages), root / "output", cap=4)
            row = next(r for r in records["post_close"] if r["ticker"] == "AAPL" and r["forecast_date"] == "2014-01-02")
            self.assertEqual((row["raw_news_count"], row["deduplicated_news_count"], row["retained_news_count"]), (13, 12, 4))
            self.assertEqual([r["id"] for r in row["headlines"]], sorted(r["id"] for r in row["headlines"]))
            # Reconstruct full cap, independently check deterministic prefix and earliest duplicate.
            full, _ = prepare(root / "source", root / "output_full", cap=64)
            fullrow = next(r for r in full["post_close"] if r["ticker"] == "AAPL" and r["forecast_date"] == "2014-01-02")
            self.assertEqual(row["headlines"], fullrow["headlines"][:4])
            self.assertEqual(next(m for m in fullrow["headlines"] if m["text"] == "headline 0")["created_at"], "2014-01-02T14:00:00Z")
            empty = next(r for r in records["pre_open"] if r["ticker"] == "AAPL" and r["target_date"] == "2014-01-06")
            self.assertEqual(empty["headlines"], [])
            self.assertEqual(empty["retained_news_count"], 0)
            # Alter the future outcome and check that the earlier news selection
            # is unchanged rather than merely claiming label independence.
            price_path = root / "source/price/raw/AAPL.csv"
            with price_path.open(newline="") as stream:
                changed_prices = list(csv.DictReader(stream))
            for price in changed_prices:
                if price["Date"] == "2014-01-03":
                    price["Close"] = price["Adj Close"] = "99"
            with price_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(changed_prices[0]))
                writer.writeheader()
                writer.writerows(changed_prices)
            changed, _ = prepare(root / "source", root / "output_changed", cap=4)
            changed_row = next(r for r in changed["post_close"] if r["id"] == row["id"])
            self.assertNotEqual(changed_row["target_up"], row["target_up"])
            self.assertEqual(changed_row["headlines"], row["headlines"])

    def test_future_unlock_is_explicit_and_existing_outputs_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root, future_bad=False)
            records, manifest = prepare(source, root / "output", include_test=True)
            self.assertTrue(any(r["split"] == "test" for r in records["post_close"]))
            self.assertIn("test", manifest["summary"]["post_close"])
            before = (root / "output/provenance.json").read_bytes()
            with self.assertRaises(FileExistsError):
                prepare(source, root / "output")
            self.assertEqual(before, (root / "output/provenance.json").read_bytes())

    def test_timestamp_parser_and_previous_session_are_strict(self):
        self.assertEqual(parse_created_at("Mon Jan 06 14:00:00 +0200 2014"), datetime(2014, 1, 6, 12, tzinfo=timezone.utc))
        with self.assertRaises(ValueError):
            parse_created_at("2014-01-06 12:00:00")
        self.assertEqual(previous_session(["2014-01-02", "2014-01-03", "2014-01-06"], "2014-01-06"), "2014-01-03")
        with self.assertRaises(ValueError):
            previous_session(["2014-01-03", "2014-01-06"], "2014-01-05")


if __name__ == "__main__":
    unittest.main()
