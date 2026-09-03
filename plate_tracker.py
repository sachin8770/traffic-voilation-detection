"""
plate_tracker.py

Solves the "many readings per car" problem in ANPR pipelines.

Instead of trusting a single frame's OCR result, this keeps a running,
confidence-weighted vote of every reading seen for a given car_id, and
lets you "finalize" (and push to your API/DB) a car once its track is
lost (i.e. it's no longer being tracked by SORT).

Usage inside your existing loop:

    tracker = PlateTracker(on_finalize=send_to_api)

    ... inside the frame loop, after you compute license_plate_text ...
    tracker.add_reading(car_id, license_plate_text, text_score, score)

    ... after the loop, or whenever `track_ids` no longer contains a car_id
    that was previously tracked ...
    tracker.check_lost_tracks(current_frame_car_ids)

    ... after the whole video ends ...
    tracker.finalize_all()
"""

import re
from collections import defaultdict


def normalize_plate(text: str) -> str:
    """Strip whitespace/punctuation and uppercase, so OCR noise like
    'MH12 AB1234' vs 'MH12AB1234' vs 'MH12-AB1234' collapse to one key."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


class PlateTracker:
    def __init__(self, on_finalize=None, min_readings: int = 2):
        """
        on_finalize: callback(car_id, best_text, confidence, num_readings)
                     called once a car's track is lost or the video ends.
        min_readings: don't finalize/send a plate seen fewer than this many
                      times — cuts down on one-off garbage reads.
        """
        self.on_finalize = on_finalize
        self.min_readings = min_readings
        # car_id -> normalized_text -> {"raw": str, "score_sum": float, "count": int}
        self._votes = defaultdict(lambda: defaultdict(lambda: {"raw": "", "score_sum": 0.0, "count": 0}))
        self._seen_car_ids = set()
        self._finalized = set()

    def add_reading(self, car_id, text, text_score, bbox_score):
        if not text:
            return
        key = normalize_plate(text)
        if not key:
            return
        combined_conf = float(text_score) * float(bbox_score)
        entry = self._votes[car_id][key]
        entry["raw"] = text if combined_conf > entry["score_sum"] / max(entry["count"], 1) else entry["raw"] or text
        entry["score_sum"] += combined_conf
        entry["count"] += 1
        self._seen_car_ids.add(car_id)

    def best_for(self, car_id):
        """Return (text, avg_confidence, num_readings) for a car's current votes."""
        candidates = self._votes.get(car_id)
        if not candidates:
            return None
        best_key = max(candidates, key=lambda k: candidates[k]["score_sum"])
        entry = candidates[best_key]
        avg_conf = entry["score_sum"] / entry["count"]
        return entry["raw"], avg_conf, entry["count"]

    def check_lost_tracks(self, current_frame_car_ids):
        """Call once per frame with the set of car_ids SORT is tracking
        *right now*. Any car_id we've seen before but isn't in this set
        anymore has left the frame -> finalize it."""
        current_frame_car_ids = set(current_frame_car_ids)
        lost = self._seen_car_ids - current_frame_car_ids - self._finalized
        for car_id in lost:
            self._finalize(car_id)

    def finalize_all(self):
        """Call at the end of the video to flush any remaining tracks."""
        for car_id in list(self._seen_car_ids - self._finalized):
            self._finalize(car_id)

    def _finalize(self, car_id):
        result = self.best_for(car_id)
        self._finalized.add(car_id)
        if result is None:
            return
        text, conf, count = result
        if count < self.min_readings:
            return
        if self.on_finalize:
            self.on_finalize(car_id, text, conf, count)
