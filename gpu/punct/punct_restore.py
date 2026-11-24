"""句読点・大文字小文字の復元（ASR 出力などの後処理）。

    python punct_restore.py <model_id_or_dir> "今日は晴れですね明日は雨でしょう"
    from punct_restore import restore; restore(model, tok, text)

入力は句読点なしの文を想定（小文字化は内部で行う）。長い文は 250 トークンごとに区切って処理する。
"""
import re, sys
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

PUNCT = ["O", "COMMA", "PERIOD", "QUESTION", "EXCLAIM", "COLON"]
OUT = {"ja": {1: "、", 2: "。", 3: "？", 4: "！", 5: "："},
       "zh": {1: "，", 2: "。", 3: "？", 4: "！", 5: "："},
       "other": {1: ",", 2: ".", 3: "?", 4: "!", 5: ":"}}


def _script(s):
    if re.search(r"[぀-ヿ]", s): return "ja"
    if re.search(r"[一-鿿]", s) and not re.search(r"[가-힯]", s): return "zh"
    return "other"


def _lower(text):
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text)


@torch.no_grad()
def restore(model, tok, text, window=250):
    text = re.sub(r"\s+", " ", text.strip())
    inp = _lower(text)
    enc = tok(inp, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    after, up = [0] * len(inp), [False] * len(inp)
    for a in range(0, len(ids), window):
        chunk = [tok.cls_token_id] + ids[a:a + window] + [tok.sep_token_id]
        x = torch.tensor([chunk], device=model.device)
        pred = model(input_ids=x).logits[0].argmax(-1).tolist()[1:-1]
        for (s, e), p in zip(offs[a:a + window], pred):
            while s < e and inp[s].isspace(): s += 1
            while e > s and inp[e - 1].isspace(): e -= 1
            if e <= s: continue
            after[e - 1] = p // 2
            if p % 2: up[s] = True
    sc, out = _script(inp), []
    for i, c in enumerate(text):          # 元の綴りを保ち、大文字化だけ反映
        out.append(c.upper() if up[i] else c)
        if after[i]: out.append(OUT[sc][after[i]])
    return "".join(out)


if __name__ == "__main__":
    name, text = sys.argv[1], " ".join(sys.argv[2:])
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForTokenClassification.from_pretrained(name).eval()
    print(restore(model, tok, text))
