"""基準 2: 1-800-BAD-CODE の punctuators (pcs_47lang, Apache-2.0, ONNX/CPU)。出力文字列を入力に位置合わせして、こちらと同じ指標で測る。"""
import glob, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import common
from punctuators.models import PunctCapSegModelONNX

N = int(os.environ.get("N", 300))
m = PunctCapSegModelONNX.from_pretrained("pcs_47lang")

def align(inp, out):
    """英数字(かな・漢字含む)の列だけで位置合わせし、句読点は直前の英数字に付ける。<unk> は捨てる。"""
    out = re.sub(r"<unk>", "", out, flags=re.I)
    idx = [i for i, c in enumerate(inp) if c.isalnum()]
    n = len(inp); after = [0] * n; up = [False] * n; k = 0; last = -1
    for ch in out:
        if ch.isalnum():
            if k < len(idx):
                pos = idx[k]; up[pos] = ch != inp[pos] and ch.lower() == inp[pos]; last = pos; k += 1
        elif ch in common.MAP and last >= 0 and after[last] == 0:
            if not (last + 1 < n and inp[last + 1] == ch):    # 入力に元からある記号は無視
                after[last] = common.MAP[ch]
    return after, up

meter = common.Meter(); t0 = time.time(); samples = []
for path in sorted(glob.glob("data/proc/test_*.jsonl")):
    key = common.file_key(path)
    for t in common.read_texts(path, N):
        inp, cased, after, upper = common.clean_example(t)
        segs = m.infer([inp])[0]
        out = " ".join(segs)
        pa, pu = align(inp, out)
        meter.add_punct(key, after, pa); meter.add_case(key, inp, upper, pu)
    if len(samples) < 40: samples.append((key, inp[:100], out[:140]))
    print(key, round(time.time() - t0), flush=True)
res = meter.summary()
allm = common.Meter()
for k, d in meter.p.items():
    for kk, v in d.items(): allm.p["ALL"][kk] += v
    allm.n["ALL"] += meter.n[k]
for k, d in meter.c.items():
    for kk, v in d.items(): allm.c["ALL"][kk] += v
res["ALL"] = allm.summary()["ALL"]
json.dump(res, open(os.environ.get("OUTJSON", "runs/015_baselines/pcs47.json"), "w"), indent=1, ensure_ascii=False)
open(os.environ.get("OUTJSON", "runs/015_baselines/pcs47.json").replace(".json", "_samples.md"), "w").write("\n".join(f"- {k}\n  - in: {a}\n  - out: {b}" for k, a, b in samples))
print({k: v["punct_micro"]["F1"] for k, v in res.items()})
