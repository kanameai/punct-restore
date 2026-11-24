"""保存済みモデルを data/proc/test_*.jsonl（全部）で評価。usage: eval_model.py <model_dir> <out_json>"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch, common
from transformers import AutoTokenizer, AutoModelForTokenClassification
d, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(d); m = AutoModelForTokenClassification.from_pretrained(d).cuda()
s, samples = common.evaluate(m, tok, sorted(glob.glob("data/proc/test_*.jsonl")), int(os.environ.get("N", 2000)), "cuda", n_samples=4)
json.dump(s, open(out, "w"), indent=1, ensure_ascii=False)
with open(out.replace(".json", "_samples.md"), "w") as f:
    for x in samples: f.write(f"## {x['key']}\n- 入力: {x['input']}\n- 予測: {x['pred']}\n- 正解: {x['gold']}\n\n")
print({k: v["punct_micro"]["F1"] for k, v in s.items()})
