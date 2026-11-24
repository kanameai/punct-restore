"""mmBERT-base で句読点(5)×大文字(2)=12 クラスのトークン分類を学習。素の PyTorch ループ（HF Trainer の API 変更を避ける）。"""
import csv, glob, json, os, sys, time, math, random
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn.functional as F, common
from transformers import AutoTokenizer, AutoModelForTokenClassification

BASE = os.environ.get("BASE", "jhu-clsp/mmBERT-base")
NAME = os.environ.get("NAME", "punct-base-v0")
EPOCHS = float(os.environ.get("EPOCHS", 2)); BS = int(os.environ.get("BS", 64)); LR = float(os.environ.get("LR", 5e-5))
MAXLEN = 256; MAX_HOURS = float(os.environ.get("MAX_HOURS", 18)); EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 2000))
AUG = float(os.environ.get("AUG", 0.0))         # 断片切り出し（ASR の区切り方を模す）の確率
MAX_STEPS = int(os.environ.get("MAX_STEPS", 0))      # >0 ならスモークテスト
OUT = f"runs/{os.environ.get('JOB', '003_train')}"; os.makedirs(OUT, exist_ok=True)
MODEL_DIR = os.path.expanduser(f"~/horizon/models/{NAME}"); os.makedirs(MODEL_DIR, exist_ok=True)
dev = "cuda"; torch.manual_seed(0); random.seed(0)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForTokenClassification.from_pretrained(BASE, num_labels=len(common.LABELS),
                                                        id2label=common.ID2LABEL, label2id=common.LABEL2ID).to(dev)
print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M")
texts = common.read_texts(os.environ.get("TRAIN_FILE", "data/proc/train.jsonl"))
if MAX_STEPS: texts = texts[: MAX_STEPS * BS]
print("train paragraphs", len(texts))

class DS(torch.utils.data.Dataset):
    def __len__(self): return len(texts)
    def __getitem__(self, i):
        t = texts[i]
        if AUG and len(t) > 60 and random.random() < AUG:
            sp = [j for j, c in enumerate(t) if c == " "]
            if len(sp) >= 6:                       # 空白区切りの言語: 語境界で切る
                a, b = sorted(random.sample(sp, 2))
                if b - a >= 30: t = t[a + 1:b]
            elif " " not in t:                     # 日本語・中国語: 任意の文字位置で切る
                a = random.randrange(0, len(t) - 30); b = random.randrange(a + 30, len(t) + 1); t = t[a:b]
        e = common.encode(tok, t, MAXLEN)
        if e is None: e = common.encode(tok, "これはテストです。", MAXLEN)
        return e["input_ids"], e["labels"]

def collate(b):
    L = max(len(x[0]) for x in b)
    ids = torch.full((len(b), L), tok.pad_token_id); lab = torch.full((len(b), L), -100); am = torch.zeros((len(b), L), dtype=torch.long)
    for i, (x, y) in enumerate(b):
        ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); am[i, :len(x)] = 1
    return ids, am, lab

dl = torch.utils.data.DataLoader(DS(), batch_size=BS, shuffle=True, num_workers=6, collate_fn=collate, drop_last=True, persistent_workers=True)
total = int(len(dl) * EPOCHS) if not MAX_STEPS else MAX_STEPS
nd = [p for n, p in model.named_parameters() if p.ndim < 2 or "norm" in n.lower() or "bias" in n]
dd = [p for n, p in model.named_parameters() if not (p.ndim < 2 or "norm" in n.lower() or "bias" in n)]
opt = torch.optim.AdamW([{"params": dd, "weight_decay": 0.01}, {"params": nd, "weight_decay": 0.0}], lr=LR, betas=(0.9, 0.98), eps=1e-6, fused=True)
warm = max(1, int(0.03 * total))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
dev_files = sorted(glob.glob("data/proc/dev_*.jsonl")); test_files = sorted(glob.glob("data/proc/test_*.jsonl"))
curve = csv.writer(open(f"{OUT}/curve.csv", "w")); curve.writerow(["step", "loss", "lr", "tokens_per_s", "elapsed_min"])
devlog = open(f"{OUT}/dev_metrics.jsonl", "w")
t0 = time.time(); step = 0; run_loss = 0.0; run_tok = 0; tl = time.time(); done = False
model.train()
while not done:
    for ids, am, lab in dl:
        ids, am, lab = ids.to(dev), am.to(dev), lab.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=am).logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), lab.view(-1), ignore_index=-100)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1; run_loss += loss.item(); run_tok += int(am.sum())
        if step % 100 == 0:
            now = time.time(); tps = run_tok / (now - tl)
            curve.writerow([step, round(run_loss / 100, 4), f"{sched.get_last_lr()[0]:.2e}", int(tps), round((now - t0) / 60, 1)])
            print(f"step {step}/{total} loss {run_loss / 100:.4f} tok/s {int(tps)} elapsed {(now - t0) / 60:.1f}min", flush=True)
            run_loss = 0.0; run_tok = 0; tl = now
        if step % EVAL_EVERY == 0 or step == total:
            s, _ = common.evaluate(model, tok, dev_files, 200, dev)
            devlog.write(json.dumps({"step": step, "ALL": s["ALL"], "per_lang_punct_f1": {k: v["punct_micro"]["F1"] for k, v in s.items()}}, ensure_ascii=False) + "\n"); devlog.flush()
            print("DEV", step, s["ALL"]["punct_micro"], s["ALL"].get("case_uc"), flush=True)
        if step % (len(dl)) == 0 or step == total:
            model.save_pretrained(MODEL_DIR); tok.save_pretrained(MODEL_DIR)
        if step >= total or (time.time() - t0) / 3600 > MAX_HOURS:
            done = True; break
model.save_pretrained(MODEL_DIR); tok.save_pretrained(MODEL_DIR)
summ, samples = common.evaluate(model, tok, test_files, 2000, dev, n_samples=3)
json.dump(summ, open(f"{OUT}/test_metrics.json", "w"), indent=1, ensure_ascii=False)
with open(f"{OUT}/samples.md", "w") as f:
    for s in samples:
        f.write(f"## {s['key']}\n- 入力: {s['input']}\n- 予測: {s['pred']}\n- 正解: {s['gold']}\n\n")
print("TEST ALL", summ["ALL"]); print("steps", step, "minutes", (time.time() - t0) / 60)
