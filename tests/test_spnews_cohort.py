import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import spnews_cohort as cohort


def news(day, *rows):
    buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=sorted(cohort.REQUIRED_NEWS))
    writer.writeheader()
    for item in rows:
        row = {key: "" for key in cohort.REQUIRED_NEWS}
        row.update(Id="x", Title="Headline", Company="A", Date=day,
                   ProviderPublishTime=str(int(datetime.fromisoformat(day).replace(hour=12, tzinfo=timezone.utc).timestamp())))
        row.update(item); writer.writerow(row)
    return buffer.getvalue().encode()


def prices():
    days = ["2022-09-19", "2022-09-20", "2022-09-21", "2022-09-22", "2022-09-23", "2022-09-26", "2022-09-27"]
    values = {day: [100, 101, 99, 100 + i, 100 + i, 1000] for i, day in enumerate(days)}
    return days, values


class TestSPNewsCohort(unittest.TestCase):
    def test_backlog_blacklist_first_version_and_original_content(self):
        snapshots = [
            ("2022-09-20", news("2022-09-20", {"Id": "back", "Title": "Old Story"})),
            ("2022-09-21", news("2022-09-21", {"Id": "back", "Title": "Revised old"},
                {"Id": "new-back-id", "Title": " old   STORY "},
                {"Id": "new", "Title": " First   version ", "Company": "A"},
                {"Id": "new", "Title": "First   version", "Company": "B"},
                {"Id": "alias", "Title": "first version", "Company": "C"},
                {"Id": "alias", "Title": "Alias revised", "Company": "D"})),
            ("2022-09-22", news("2022-09-22", {"Id": "new", "Title": "Later revision", "Company": "E"},
                {"Id": "different", "Title": "First version", "Company": "F"})),
        ]
        stories, stats = cohort.parse_stories(snapshots)
        self.assertEqual(len(stories), 1)
        story = stories[0]
        self.assertEqual(story["text"], "First   version")
        self.assertEqual(story["id"], hashlib.sha256(story["text"].encode()).hexdigest())
        self.assertEqual(story["company_links"], ["A", "B", "C"])
        self.assertEqual(story["first_snapshot_date"], "2022-09-21")
        self.assertEqual(story["available_at"], "2022-09-23T00:00:00Z")
        self.assertEqual(stats["totals"]["excluded_backlog_reappearances"], 2)

    def test_fatal_schema_date_and_future_clock_guards(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            cohort.parse_stories([("2022-09-20", b"Id,Title\nx,Hello\n")])
        with self.assertRaisesRegex(ValueError, "Date/filename"):
            cohort.parse_stories([("2022-09-20", news("2022-09-21", {}))])
        future = int(datetime(2022, 9, 24, tzinfo=timezone.utc).timestamp())
        with self.assertRaisesRegex(ValueError, "publication beyond"):
            cohort.parse_stories([("2022-09-20", news("2022-09-20", {"ProviderPublishTime": future}))])
        stories, stats = cohort.parse_stories([("2022-09-20", news("2022-09-20")),
            ("2022-09-21", news("2022-09-21", {"Id": "bad", "ProviderPublishTime": "nan"},
                {"Id": "header", "Title": "Title"}, {"Id": "empty", "Title": ""}))])
        self.assertEqual(stories, [])
        self.assertEqual(stats["totals"]["rejected_invalid_publication_epoch"], 1)
        self.assertEqual(stats["totals"]["rejected_header_or_encoding"], 1)

    def test_prices_future_cells_never_converted_and_adjusted_label(self):
        raw = b"Date,Open,High,Low,Close,AdjClose,Volume\n2023-12-27,200,205,195,200,100,1000\n2023-12-28,100,102,99,100,101,1000\n2023-12-29,100,102,99,100,102,1000\n2024-01-02,DO_NOT_PARSE,NO,NO,NO,NO,NO\n"
        calendar, values = cohort.read_prices(raw)
        self.assertNotIn("2024-01-02", values)
        rows = cohort.build_cohort([], calendar, values, start="2023-12-28")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target_up"], 1)
        self.assertEqual(rows[0]["price_date"], "2023-12-28")
        self.assertEqual(rows[0]["target_date"], "2023-12-29")

    def test_clock_windows_empty_days_and_history_outcome_allowlist(self):
        from src.finin_history_data import HistoryCohort
        stories, _ = cohort.parse_stories([("2022-09-20", news("2022-09-20")),
            ("2022-09-21", news("2022-09-21", {"Id": "new", "Title": "Public headline"}))])
        calendar, values = prices()
        primary = cohort.build_cohort(stories, calendar, values)
        retrospective = cohort.build_cohort(stories, calendar, values, policy="publication_only_retrospective")
        self.assertEqual([row["id"] for row in primary], [row["id"] for row in retrospective])
        self.assertEqual(len(primary), 5)
        self.assertEqual([row["price_date"] for row in primary if row["headlines"]], ["2022-09-23"])
        self.assertEqual([row["price_date"] for row in retrospective if row["headlines"]], ["2022-09-21"])
        self.assertTrue(all("target_up" in row for row in primary if not row["headlines"]))
        h = HistoryCohort(primary, calendar, history_days=3, development_target_before=cohort.BOUNDARY)
        self.assertEqual(len(h), 3)
        self.assertTrue(all("target_up" not in row and "target_return" not in row for history in h.histories for row in history))
        self.assertEqual(h.histories[-1][1]["headlines"][0]["available_at"], "2022-09-23T00:00:00Z")
        del values["2022-09-22"]
        with self.assertRaisesRegex(ValueError, "Missing development price session"):
            cohort.build_cohort(stories, calendar, values)

    def test_hash_cap_is_label_and_input_order_independent(self):
        stories, _ = cohort.parse_stories([("2022-09-20", news("2022-09-20")),
            ("2022-09-21", news("2022-09-21", *[{"Id": str(i), "Title": f"Headline {i}"} for i in range(100)]))])
        calendar, values = prices()
        first = cohort.build_cohort(stories, calendar, values)
        values["2022-09-26"][4] = 50
        second = cohort.build_cohort(list(reversed(stories)), calendar, values)
        a = next(row for row in first if row["headlines"])
        b = next(row for row in second if row["headlines"])
        self.assertNotEqual(a["target_up"], b["target_up"])
        self.assertEqual(a["headlines"], b["headlines"])
        self.assertEqual(a["news_count_precap"], 100)
        self.assertEqual(len(a["headlines"]), 64)

    def test_missing_collection_is_distinct_from_observed_empty_and_weekends(self):
        calendar, values = prices()
        observed = ["2022-09-18", "2022-09-19", "2022-09-20", "2022-09-21", "2022-09-22", "2022-09-24"]
        rows = cohort.build_cohort([], calendar, values, snapshot_dates=observed)
        weekday = next(row for row in rows if row["price_date"] == "2022-09-23")
        monday = next(row for row in rows if row["price_date"] == "2022-09-26")
        self.assertEqual(weekday["news_absence_status"], "no_new_eligible_stories_in_observed_snapshots")
        self.assertEqual(monday["proxy_window_snapshot_dates_expected"], ["2022-09-22", "2022-09-23", "2022-09-24"])
        self.assertEqual(monday["proxy_window_snapshot_dates_missing"], ["2022-09-23"])
        self.assertFalse(monday["collection_window_complete"])
        self.assertEqual(monday["news_absence_status"], "no_eligible_stories_collection_incomplete")
        self.assertTrue(monday["collection_missing"])
        self.assertEqual(monday["missing_collection_dates"], ["2022-09-23"])
        self.assertFalse(monday["observed_empty_news"])
        self.assertTrue(weekday["observed_empty_news"])
        retrospective = cohort.build_cohort([], calendar, values, snapshot_dates=observed, policy="publication_only_retrospective")
        self.assertTrue(all(row["coverage_status"] == "unverified_publication_vintage" and not row["observed_empty_news"] for row in retrospective))
        self.assertEqual(len(rows), 5)
        self.assertEqual(cohort._date_runs(["2023-03-07", "2023-03-08", "2023-03-20"]),
                         [{"start": "2023-03-07", "end": "2023-03-08", "days": 2}, {"start": "2023-03-20", "end": "2023-03-20", "days": 1}])

    def test_resumable_pinned_commit_and_mutation_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            audit, out = Path(temp)/"audit", Path(temp)/"cohort"
            audit.mkdir()
            samples = {cohort.PRICE: b"Date,Open,High,Low,Close,AdjClose,Volume\n2022-09-19,1,1,1,1,1,1\n2022-09-20,1,1,1,1,1,1\n2022-09-21,1,1,1,1,2,1\n",
                       cohort.DAILY+cohort.START+".csv": news(cohort.START, {})}
            tree = {"truncated": False, "tree": [{"path": p, "type": "blob", "size": len(raw), "sha": cohort._blob(raw)} for p, raw in samples.items()]}
            tree_bytes = json.dumps(tree).encode(); (audit/"source_tree.json").write_bytes(tree_bytes)
            (audit/"manifest.json").write_text(json.dumps({"revision": cohort.REVISION, "source_tree_sha256": cohort._sha(tree_bytes)}))
            for remote, raw in samples.items():
                local = audit/"sources"/remote; local.parent.mkdir(parents=True, exist_ok=True); local.write_bytes(raw)
            with patch.object(cohort, "_get", side_effect=AssertionError("must reuse audit, no network")):
                result = cohort.run(out, audit, workers=1)
                self.assertEqual(result["network_download_bytes"], 0)
                self.assertEqual(cohort.run(out, audit, workers=1), result)
            (out/"primary/devsamples_post_close.jsonl").write_text("modified")
            with self.assertRaisesRegex(ValueError, "Prepared output checksum"):
                cohort.run(out, audit, workers=1)


if __name__ == "__main__": unittest.main()
