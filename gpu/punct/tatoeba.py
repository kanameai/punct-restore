"""Tatoeba の CC0 文（sentences_CC0.tar.bz2）→ data/raw/tatoeba_<lang>.jsonl。1〜3 文をつないだ発話単位にする。標準ライブラリのみ。"""
import bz2, io, json, os, random, tarfile, urllib.request
URL = os.environ.get("TATOEBA_URL", "https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2")
JOB = os.environ.get("JOB", "005_tatoeba")
LANG = {"jpn": "ja", "eng": "en", "deu": "de", "fra": "fr", "spa": "es", "ita": "it", "por": "pt", "nld": "nl", "rus": "ru", "cmn": "zh", "kor": "ko"}
CAP = 40000
os.makedirs("data/tatoeba", exist_ok=True); os.makedirs("data/raw", exist_ok=True)
tgz = "data/tatoeba/" + URL.rsplit("/", 1)[1]
if not os.path.exists(tgz):
    urllib.request.urlretrieve(URL, tgz)
by = {v: [] for v in LANG.values()}
with tarfile.open(tgz, "r:bz2") as t:
    for m in t.getmembers():
        if not m.isfile(): continue
        for line in io.TextIOWrapper(t.extractfile(m), encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[1] in LANG and 8 <= len(parts[2]) <= 300:
                by[LANG[parts[1]]].append((parts[0], parts[2].strip()))
random.seed(1); stats = {}
for lg, rows in by.items():
    random.shuffle(rows); out = []; i = 0
    while i < len(rows) and len(out) < CAP:
        k = random.choice([1, 1, 2, 3]); grp = rows[i:i + k]; i += k
        sep = "" if lg in ("ja", "zh") else " "
        out.append({"lang": lg, "src": "tatoeba", "title": grp[0][0], "text": sep.join(t for _, t in grp)})
    with open(f"data/raw/tatoeba_{lg}.jsonl", "w") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats[lg] = {"sentences_available": len(rows), "groups": len(out)}
os.makedirs(f"runs/{JOB}", exist_ok=True)
json.dump(stats, open(f"runs/{JOB}/stats.json", "w"), indent=1)
for lg in by: os.system(f"head -n 4 data/raw/tatoeba_{lg}.jsonl > runs/{JOB}/sample_{lg}.jsonl")
print(stats)
