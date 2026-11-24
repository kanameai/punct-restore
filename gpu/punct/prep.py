"""data/raw → data/proc（学習/開発/テスト分割）＋ トークナイザの天井・例の確認。"""
import glob, hashlib, json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
import common
from transformers import AutoTokenizer

OUT = "runs/" + os.environ.get("JOB", "002_prep"); os.makedirs(OUT, exist_ok=True); os.makedirs("data/proc", exist_ok=True)
random.seed(0)
tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
print("tokenizer", type(tok).__name__, "fast:", tok.is_fast, "vocab", len(tok), "pad", tok.pad_token_id)

train, train_nowiki, stats = [], [], {}
for path in sorted(glob.glob("data/raw/*.jsonl")):
    key = os.path.basename(path)[:-6].replace("wiki_", "wikipedia_")
    rows = [json.loads(l) for l in open(path)]
    seen, test, dev, tr = set(), [], [], []
    for r in rows:
        t = r["text"]
        if t in seen: continue
        seen.add(t)
        if len(common.clean_example(t)[0]) < 20: continue
        h = int(hashlib.md5((r["src"] + r["title"]).encode()).hexdigest(), 16) % 100
        (test if h < 2 else dev if h < 3 else tr).append(t)
    random.shuffle(test); random.shuffle(dev)
    test, dev = test[:2000], dev[:300]
    for name, lst in (("test", test), ("dev", dev)):
        with open(f"data/proc/{name}_{key}.jsonl", "w") as f:
            for t in lst: f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    train += tr
    if not key.startswith("wikipedia"): train_nowiki += tr
    stats[key] = {"train": len(tr), "dev": len(dev), "test": len(test)}
random.shuffle(train)
with open("data/proc/train.jsonl", "w") as f:
    for t in train: f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
random.shuffle(train_nowiki)
with open("data/proc/train_nowiki.jsonl", "w") as f:       # Wikipedia(CC-BY-SA) を使わない学習用
    for t in train_nowiki: f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
stats["_train_total"] = len(train); stats["_train_nowiki_total"] = len(train_nowiki)

# 天井: 正解ラベルをトークン位置に載せて戻したとき、句読点がどれだけ残るか。長さ・ラベル分布も。
meter = common.Meter(); lens = {}; trunc = {}; dist = {}
examples = []
for path in sorted(glob.glob("data/proc/test_*.jsonl")):
    key = common.file_key(path); L = []; tr = 0; cnt = [0] * 6
    for i, t in enumerate(common.read_texts(path, 500)):
        e = common.encode(tok, t, 256)
        if not e: continue
        L.append(len(e["input_ids"])); tr += int(len(e["input_ids"]) >= 256)
        for a in e["after"]: cnt[a] += 1
        gold_ids = [l if l != -100 else 0 for l in e["labels"]]
        pa, pu = common.decode(e["spans"], gold_ids, len(e["inp"]))
        n = len(e["inp"]); m = n
        meter.add_punct(key, e["after"], pa); meter.add_case(key, e["inp"], e["upper"], pu)
        if i < 2:
            examples.append((key, t[:160], e["inp"][:160], common.render(e["inp"], pa, pu)[:170],
                             tok.convert_ids_to_tokens(e["input_ids"])[:24]))
    lens[key] = {"mean_tokens": round(sum(L) / max(1, len(L)), 1), "max": max(L) if L else 0}
    trunc[key] = round(tr / max(1, len(L)), 4); dist[key] = dict(zip(common.PUNCT, cnt))
ceil = meter.summary()
json.dump({"split": stats, "token_len": lens, "trunc_rate_256": trunc, "label_counts_per_500par": dist},
          open(f"{OUT}/data_stats.json", "w"), indent=1, ensure_ascii=False)
json.dump(ceil, open(f"{OUT}/token_ceiling.json", "w"), indent=1, ensure_ascii=False)
with open(f"{OUT}/examples.md", "w") as f:
    for key, orig, inp, back, toks in examples:
        f.write(f"## {key}\n- 元: {orig}\n- 入力(小文字・句読点なし): {inp}\n- 正解ラベルから復元: {back}\n- tokens: {toks}\n\n")
print(json.dumps(stats, ensure_ascii=False)[:600])
print({k: v["punct_micro"] for k, v in ceil.items()})
open("data/proc/READY", "w").write("ok\n")
