"""句読点・大文字小文字復元の共通部品（ラベル生成・符号化・復号・評価・描画）。"""
import json, re
from collections import defaultdict

PUNCT = ["O", "COMMA", "PERIOD", "QUESTION", "EXCLAIM", "COLON"]
CASE = ["lc", "uc"]
LABELS = [f"{p}|{c}" for p in PUNCT for c in CASE]      # id = punct*2 + case
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
LABEL2ID = {l: i for i, l in ID2LABEL.items()}

MAP = {",": 1, "、": 1, "，": 1, "､": 1, ".": 2, "。": 2, "．": 2, "｡": 2,
       "?": 3, "？": 3, "!": 4, "！": 4, ":": 5, ";": 5, "：": 5, "；": 5}
ASCII_P = set(",.?!:;")
CLOSERS = set("\"')]”’）」』")


def clean_example(text):
    """元の文から句読点を除き、(小文字化した入力, 大文字のまま, after[], upper[]) を返す。
    after[i] = 文字 i の直後にあった句読点クラス(0=なし)。upper[i] = 文字 i が大文字だったか。"""
    chars, after = [], []
    n = len(text)
    for i, c in enumerate(text):
        if c in MAP:
            ok = True
            if c in ASCII_P:
                nxt = text[i + 1] if i + 1 < n else " "
                prv = text[i - 1] if i > 0 else " "
                ok = (nxt.isspace() or nxt in CLOSERS) and prv != c and nxt != c
            if ok:
                if chars and not chars[-1].isspace() and after[-1] == 0:
                    after[-1] = MAP[c]
                continue                    # 先頭・連続・空白直後の句読点は捨てる
        if c.isspace():
            if not chars or chars[-1] == " ":
                continue
            c = " "
        chars.append(c); after.append(0)
    while chars and chars[-1] == " ":
        chars.pop(); after.pop()
    cased = "".join(chars)
    low, upper = [], []
    for c in chars:
        lo = c.lower()
        if len(lo) != 1:
            lo = c
        low.append(lo); upper.append(c != lo and c.isupper())
    return "".join(low), cased, after, upper


def encode(tok, text, max_len=256):
    inp, cased, after, upper = clean_example(text)
    if len(inp) < 5:
        return None
    enc = tok(inp, return_offsets_mapping=True, truncation=True, max_length=max_len)
    labels, spans = [], []
    for s, e in enc["offset_mapping"]:
        while s < e and inp[s].isspace(): s += 1
        while e > s and inp[e - 1].isspace(): e -= 1
        if e <= s:
            labels.append(-100); spans.append(None); continue
        labels.append(after[e - 1] * 2 + (1 if upper[s] else 0)); spans.append((s, e))
    return {"input_ids": enc["input_ids"], "labels": labels, "spans": spans,
            "inp": inp, "cased": cased, "after": after, "upper": upper}


def decode(spans, pred_ids, n):
    after, up = [0] * n, [False] * n
    for sp, p in zip(spans, pred_ids):
        if sp is None: continue
        s, e = sp
        after[e - 1] = p // 2
        if p % 2: up[s] = True
    return after, up


def script_of(s):
    if re.search(r"[぀-ヿ]", s): return "ja"
    if re.search(r"[一-鿿]", s) and not re.search(r"[가-힯]", s): return "zh"
    return "other"

OUT = {"ja": {1: "、", 2: "。", 3: "？", 4: "！", 5: "："},
       "zh": {1: "，", 2: "。", 3: "？", 4: "！", 5: "："},
       "other": {1: ",", 2: ".", 3: "?", 4: "!", 5: ":"}}


def render(inp, after, up):
    sc = script_of(inp); out = []
    for i, c in enumerate(inp):
        out.append(c.upper() if up[i] else c)
        if after[i]: out.append(OUT[sc][after[i]])
    return "".join(out)


class Meter:
    def __init__(self):
        self.p = defaultdict(lambda: defaultdict(int))   # key -> "tp1","fp1","fn1",...
        self.c = defaultdict(lambda: defaultdict(int))
        self.n = defaultdict(int)

    def add_punct(self, key, gold, pred):
        d = self.p[key]; self.n[key] += len(gold)
        for g, p in zip(gold, pred):
            if g == p:
                if g: d[f"tp{g}"] += 1
            else:
                if p: d[f"fp{p}"] += 1
                if g: d[f"fn{g}"] += 1

    def add_case(self, key, inp, gold_up, pred_up):
        d = self.c[key]
        for i, ch in enumerate(inp):
            if (i == 0 or inp[i - 1].isspace()) and ch.upper() != ch:
                g, p = gold_up[i], pred_up[i]
                if g and p: d["tp"] += 1
                elif p: d["fp"] += 1
                elif g: d["fn"] += 1
                else: d["tn"] += 1

    @staticmethod
    def _prf(tp, fp, fn):
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        return {"P": round(P, 4), "R": round(R, 4), "F1": round(2 * P * R / (P + R), 4) if P + R else 0.0,
                "support": tp + fn}

    def summary(self):
        res = {}
        for key in self.p:
            d = self.p[key]; r = {"chars": self.n[key]}
            T = [0, 0, 0]
            for k in range(1, 6):
                tp, fp, fn = d[f"tp{k}"], d[f"fp{k}"], d[f"fn{k}"]
                r[PUNCT[k]] = self._prf(tp, fp, fn); T[0] += tp; T[1] += fp; T[2] += fn
            r["punct_micro"] = self._prf(*T)
            cd = self.c.get(key)
            if cd and (cd["tp"] + cd["fn"] + cd["fp"] + cd["tn"]):
                r["case_uc"] = self._prf(cd["tp"], cd["fp"], cd["fn"])
                r["case_acc"] = round((cd["tp"] + cd["tn"]) / (cd["tp"] + cd["fp"] + cd["fn"] + cd["tn"]), 4)
            res[key] = r
        return res


def read_texts(path, n=None):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line)["text"])
            if n and len(out) >= n: break
    return out


def file_key(path):
    import os
    b = os.path.basename(path)
    return b.replace("test_", "").replace("dev_", "").replace(".jsonl", "")


def evaluate(model, tok, files, n_per_file, device, bs=64, max_len=256, n_samples=3):
    import torch
    meter = Meter(); samples = []
    model.eval()
    for path in files:
        key = file_key(path)
        exs = [e for e in (encode(tok, t, max_len) for t in read_texts(path, n_per_file)) if e]
        for b in range(0, len(exs), bs):
            batch = exs[b:b + bs]
            L = max(len(e["input_ids"]) for e in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id); am = torch.zeros((len(batch), L), dtype=torch.long)
            for i, e in enumerate(batch):
                ids[i, :len(e["input_ids"])] = torch.tensor(e["input_ids"]); am[i, :len(e["input_ids"])] = 1
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids.to(device), attention_mask=am.to(device)).logits
            pred = logits.argmax(-1).cpu().tolist()
            for i, e in enumerate(batch):
                n = len(e["inp"]); pa, pu = decode(e["spans"], pred[i][:len(e["spans"])], n)
                meter.add_punct(key, e["after"], pa); meter.add_case(key, e["inp"], e["upper"], pu)
                if b == 0 and i < n_samples:
                    samples.append({"key": key, "input": e["inp"][:200], "pred": render(e["inp"], pa, pu)[:240],
                                    "gold": render(e["inp"], e["after"], e["upper"])[:240]})
    # 全体
    summ = meter.summary()
    allm = Meter()
    for k, d in meter.p.items():
        for kk, v in d.items(): allm.p["ALL"][kk] += v
        allm.n["ALL"] += meter.n[k]
    for k, d in meter.c.items():
        for kk, v in d.items(): allm.c["ALL"][kk] += v
    summ["ALL"] = allm.summary()["ALL"]
    model.train()
    return summ, samples
