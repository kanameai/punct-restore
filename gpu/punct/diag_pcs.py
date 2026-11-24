"""punctuators 評価の健全性チェック: 出力が入力を欠落なく再現しているか（位置合わせの前提）と、全文の例。"""
import glob, os, re, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import common
from punctuators.models import PunctCapSegModelONNX
m = PunctCapSegModelONNX.from_pretrained("pcs_47lang")
rep = {}
for key in ["wikipedia_en", "wikipedia_de", "wikipedia_ja", "tatoeba_en", "tatoeba_ja"]:
    texts = common.read_texts(f"data/proc/test_{key}.jsonl", 60)
    cov, sync, nseg, ex = [], [], [], []
    for t in texts:
        inp, cased, after, upper = common.clean_example(t)
        segs = m.infer([inp])[0]; out = " ".join(segs)
        a = [c for c in inp if c.isalnum()]; b = [c.lower() for c in re.sub(r"<unk>", "", out, flags=re.I) if c.isalnum()]
        cov.append(len(b) / max(1, len(a)))
        sync.append(sum(x == y for x, y in zip(a, b)) / max(1, len(a)))
        nseg.append(len(segs))
        if len(ex) < 2: ex.append({"len_in": len(inp), "inp": inp, "out": out, "gold": common.render(inp, after, upper), "n_segments": len(segs)})
    rep[key] = {"mean_alnum_coverage": round(sum(cov) / len(cov), 3), "min_coverage": round(min(cov), 3),
                "mean_sync": round(sum(sync) / len(sync), 3), "mean_segments": round(sum(nseg) / len(nseg), 2), "examples": ex}
    print(key, {k: v for k, v in rep[key].items() if k != "examples"}, flush=True)
os.makedirs("runs/017_diag_pcs", exist_ok=True)
json.dump(rep, open("runs/017_diag_pcs/diag.json", "w"), indent=1, ensure_ascii=False)
