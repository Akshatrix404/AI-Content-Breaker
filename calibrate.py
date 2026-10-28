"""
Empirical calibration (no ML): run the scorer on every passage of data/human_corpus.txt and record,
for each of the 17 features, the MEDIAN raw score that genuine human writing receives.
At scoring time each feature is rescaled so that the human median -> 0% and 100% stays 100%:
    calibrated = clip((raw - human_median) / (1 - human_median))
Run again whenever you add passages:  python calibrate.py
"""
import json, os, statistics as st
from pipeline import analyze_text

HERE = os.path.dirname(__file__)
os.environ["AICB_NO_CAL"] = "1"
docs = [d.strip() for d in open(os.path.join(HERE, "data", "human_corpus.txt"), encoding="utf-8").read().split("\n\n") if d.strip()]
raw = {}
tot = []
for d in docs:
    r = analyze_text(d)
    if "error" in r:
        continue
    tot.append(r["report"]["percentage"])
    for b in r["report"]["breakdown"]:
        raw.setdefault(b["key"], []).append(b["score_pct"] / 100.0)
cal = {k: round(st.median(v), 3) for k, v in raw.items()}
json.dump({"human_median": cal, "n_passages": len(tot), "uncalibrated_human_mean_pct": round(st.mean(tot), 1)},
          open(os.path.join(HERE, "data", "calibration.json"), "w"), indent=1)
print(len(tot), "passages; uncalibrated human mean score", round(st.mean(tot), 1))
print(cal)
