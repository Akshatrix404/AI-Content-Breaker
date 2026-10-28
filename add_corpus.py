"""Add human text from a Kaggle CSV / txt file to the corpus, then recalibrate.
   python add_corpus.py reviews.csv text_column      (CSV)
   python add_corpus.py essays.txt                    (plain text)
Good Kaggle sources: IMDB reviews, Reddit/forum posts, student essays (human-written ones), blog authorship corpus."""
import csv, re, sys, random, os
random.seed(1)
path, col = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
raw = []
if path.endswith(".csv"):
    csv.field_size_limit(10**8)
    raw = [r[col] for r in csv.DictReader(open(path, encoding="utf-8", errors="ignore")) if r.get(col)]
else:
    raw = open(path, encoding="utf-8", errors="ignore").read().split("\n\n")
out = []
for t in raw:
    ss = re.split(r"(?<=[.!?])\s+", re.sub(r"<[^>]+>|\s+", " ", t).strip())
    for i in range(0, len(ss) - 5, 5):
        c = " ".join(ss[i:i + random.randint(4, 6)])
        if 55 <= len(c.split()) <= 150 and not re.search(r"http|[^\x00-\x7F]", c):
            out.append(c)
            break
random.shuffle(out)
with open(os.path.join(os.path.dirname(__file__), "data", "human_corpus.txt"), "a", encoding="utf-8") as f:
    f.write("\n\n" + "\n\n".join(out[:1500]))
print("added", min(len(out), 1500), "passages; recalibrating...")
os.system(f'"{sys.executable}" calibrate.py')
