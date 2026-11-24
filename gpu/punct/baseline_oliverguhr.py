"""基準: oliverguhr/fullstop-punctuation-multilang-large (MIT; en/de/fr/it/nl)。大文字小文字は保ったまま句読点だけ消した入力を渡す（相手に有利）。"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch, common
from transformers import AutoTokenizer, AutoModelForTokenClassification

name = "oliverguhr/fullstop-punctuation-multilang-large"
tok = AutoTokenizer.from_pretrained(name); model = AutoModelForTokenClassification.from_pretrained(name).cuda().eval()
print(model.config.id2label)
M = {"0": 0, ",": 1, ".": 2, "?": 3, ":": 5}          # '-' は 0 扱い
res = {}
for mode in ("last_subtoken", "first_subtoken"):
    meter = common.Meter()
    for lang in ["en", "de", "fr", "it", "nl"]:
      for pre in ("wikipedia", "tatoeba"):
        path = f"data/proc/test_{pre}_{lang}.jsonl"
        if not os.path.exists(path): continue
        for t in common.read_texts(path, int(os.environ.get("N", 500))):
            inp, cased, after, upper = common.clean_example(t)
            spans = [m.span() for m in re.finditer(r"\S+", cased)]
            words = [cased[s:e] for s, e in spans]
            if not words: continue
            enc = tok(words, is_split_into_words=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad(): lg = model(**{k: v.cuda() for k, v in enc.items()}).logits[0].argmax(-1).cpu().tolist()
            wid = enc.word_ids(); lab = {}
            for ti, w in enumerate(wid):
                if w is None: continue
                if mode == "first_subtoken": lab.setdefault(w, lg[ti])
                else: lab[w] = lg[ti]
            pa = [0] * len(cased)
            for w, l in lab.items():
                pa[spans[w][1] - 1] = M.get(model.config.id2label[l], 0)
            meter.add_punct(f"{pre}_{lang}", after, pa)
    res[mode] = meter.summary()
json.dump(res, open(os.environ.get("OUTJSON", "runs/002_prep/baseline_oliverguhr.json"), "w"), indent=1)
for mode, r in res.items():
    print(mode, {k: v["punct_micro"] for k, v in r.items()})
