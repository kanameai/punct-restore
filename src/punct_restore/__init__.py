"""punct-restore: restore punctuation and capitalization in unpunctuated text (ASR output).

    from punct_restore import PunctRestorer
    pr = PunctRestorer()                    # downloads kanameaijp/punct-restore-multilingual-small
    pr("what time does the train to berlin leave")
    # 'What time does the train to Berlin leave?'
"""
import re

__version__ = "0.1.0"
__all__ = ["PunctRestorer", "restore", "MODELS", "__version__"]

MODELS = {
    "small": "kanameaijp/punct-restore-multilingual-small",
    "base": "kanameaijp/punct-restore-multilingual-base",
}
PUNCT = ["O", "COMMA", "PERIOD", "QUESTION", "EXCLAIM", "COLON"]
OUT = {"ja": {1: "、", 2: "。", 3: "？", 4: "！", 5: "："},
       "zh": {1: "，", 2: "。", 3: "？", 4: "！", 5: "："},
       "other": {1: ",", 2: ".", 3: "?", 4: "!", 5: ":"}}


def _script(s):
    if re.search(r"[\u3040-\u30ff]", s): return "ja"
    if re.search(r"[\u4e00-\u9fff]", s) and not re.search(r"[\uac00-\ud7af]", s): return "zh"
    return "other"


def _lower(text):
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text)


def restore(model, tok, text, window=250):
    """Restore punctuation in one text with an already loaded model and tokenizer."""
    import torch
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return text
    inp = _lower(text)
    enc = tok(inp, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    after, up = [0] * len(inp), [False] * len(inp)
    with torch.no_grad():
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
    for i, c in enumerate(text):   # keep the original spelling; only add capitals and marks
        out.append(c.upper() if up[i] else c)
        if after[i]: out.append(OUT[sc][after[i]])
    return "".join(out)


class PunctRestorer:
    """Load a punct-restore model once and call it on strings.

    model: "small" (default, 141M), "base" (308M), a Hugging Face model id, or a local directory.
    device: "cpu", "cuda", ... (default: cuda if available, else cpu).
    """

    def __init__(self, model="small", device=None):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        name = MODELS.get(model, model)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForTokenClassification.from_pretrained(name).to(device).eval()

    def __call__(self, text):
        if isinstance(text, str):
            return restore(self.model, self.tokenizer, text)
        return [restore(self.model, self.tokenizer, t) for t in text]

    restore = __call__
