"""モデルカード生成（評価表は gpu_results の JSON から機械的に作る）。usage: python3 make_cards.py"""
import json, os, re
R = os.path.expanduser("~/work/horizon/gpu_results/runs/")
J = lambda p: json.load(open(R + p))
f3 = lambda x: f"{x:.3f}"

V = {   # variant -> metrics
 "nowiki": {"base": ("019_train_base_nowiki", "022_eval_nowiki/base_nowiki_n300.json"),
            "small": ("021_train_small_nowiki", "022_eval_nowiki/small_nowiki_n300.json")},
 "wiki":   {"base": ("013_train_base_v2", "015_baselines/ours_base_v2_n300.json"),
            "small": ("014_train_small_v2", "015_baselines/ours_small_v2_n300.json")}}
og = J("015_baselines/oliverguhr.json")["last_subtoken"]; pc = J("016_baseline_pcs_fix/pcs47.json")

def minutes(job):
    m = re.findall(r"minutes ([0-9.]+)", open(R + job + ".log").read()); return float(m[-1])

TPL = open(os.path.join(os.path.dirname(__file__), "card_template.md")).read()

def build(variant, size):
    job, n300 = V[variant][size]; t = J(job + "/test_metrics.json"); n = J(n300)
    other = V[variant]["small" if size == "base" else "base"]
    t_o = J(other[0] + "/test_metrics.json"); n_o = J(other[1])
    nb, ns = (n, n_o) if size == "base" else (n_o, n); tb, ts = (t, t_o) if size == "base" else (t_o, t)
    wiki = variant == "wiki"; mark = lambda k: " †" if (not wiki and k.startswith("wikipedia")) else ""
    rows = ["| test set | chars | base micro-F1 | small micro-F1 | base cap-F1 |", "|---|---|---|---|---|"]
    for k in tb:
        if k == "ALL": continue
        cu = tb[k].get("case_uc"); c = cu["F1"] if cu and cu["support"] >= 300 else None
        rows.append(f"| {k}{mark(k)} | {tb[k]['chars']:,} | {f3(tb[k]['punct_micro']['F1'])} | {f3(ts[k]['punct_micro']['F1'])} | {f3(c) if c is not None else '–'} |")
    rows.append(f"| **ALL** | {tb['ALL']['chars']:,} | **{f3(tb['ALL']['punct_micro']['F1'])}** | **{f3(ts['ALL']['punct_micro']['F1'])}** | **{f3(tb['ALL']['case_uc']['F1'])}** |")
    cls = ["| class | P | R | F1 | support |", "|---|---|---|---|---|"]
    for c in ["COMMA", "PERIOD", "QUESTION", "EXCLAIM", "COLON"]:
        v = tb["ALL"][c]; cls.append(f"| {c} | {v['P']:.3f} | {v['R']:.3f} | {v['F1']:.3f} | {v['support']:,} |")
    cmp_ = ["| test set (first 300 paragraphs) | ours base | ours small | oliverguhr (MIT) | pcs_47lang (Apache-2.0) |", "|---|---|---|---|---|"]
    for k in nb:
        if k == "ALL": continue
        o = og.get(k)
        cmp_.append(f"| {k}{mark(k)} | {f3(nb[k]['punct_micro']['F1'])} | {f3(ns[k]['punct_micro']['F1'])} | {f3(o['punct_micro']['F1']) if o else '–'} | {f3(pc[k]['punct_micro']['F1'])} |")
    name = f"punct-restore-multilingual-{size}" + ("-wiki" if wiki else "")
    d = {"SIZE": size, "NAME": name, "BASE": size, "PARAMS": "308M" if size == "base" else "141M",
         "LR": "5e-5" if size == "base" else "1e-4", "EPOCHS": "2" if wiki else "3", "HOURS": f"about {minutes(job)/60:.1f} h",
         "MAIN": "\n".join(rows), "CLS": "\n".join(cls), "CMP": "\n".join(cmp_),
         "SIZE_MB": "1173" if size == "base" else "537"}
    d["DATA_TABLE"] = DATA_WIKI if wiki else DATA_NOWIKI
    d["DATA_NOTE"] = ("" if wiki else "**Wikipedia is deliberately not used for training** (license simplicity). Rows marked † are Wikipedia test sets: out-of-domain for this model "
                      "(long formal paragraphs), so scores there are clearly lower than for the Wikipedia-trained sibling. Spoken-style / short sentences are about as good as with Wikipedia.")
    d["LICENSE_NOTE"] = (LIC_WIKI.replace("{size}", size) if wiki else LIC_NOWIKI)
    d["EVAL_NOTE"] = ("All test sets come from the same sources as the training data" if wiki else
                      "Tatoeba and Aozora test sets come from the same sources as the training data (held out by sentence-group / work), so these are optimistic for real-world input such as ASR transcripts; the Wikipedia sets (†) are not in the training data")
    d["CMP_NOTE"] = ("" if wiki else (
        "On the Wikipedia rows (†), which are not in this model's training data, the base model is roughly on par with oliverguhr (better in de/nl/en, slightly worse in fr/it); the small model is better in de/nl but worse in en/fr/it. The large margins elsewhere come from matching training data, not from a stronger model in general."))
    d["JA_LIMIT"] = ("Japanese: written Wikipedia / Tatoeba / literary style (Aozora Bunko)." if wiki else
                     "Japanese: Tatoeba sentences and literary style (Aozora Bunko); no Wikipedia in training.")
    out = TPL
    for k, v in d.items(): out = out.replace("__" + k + "__", v)
    return name, out

DATA_NOWIKI = """| source | license | use |
|---|---|---|
| Aozora Bunko, cleaned (`globis-university/aozorabunko-clean`) | CC BY 4.0 (texts are public domain) | ~58k paragraphs (Japanese, mostly pre-war literary style) |
| [Tatoeba](https://tatoeba.org) sentences | CC BY 2.0 FR — attribution: Tatoeba contributors | up to 40k utterances per language, 11 languages (1–3 sentences joined) |"""
DATA_WIKI = """| source | license | use |
|---|---|---|
| Wikipedia, 11 languages (`wikimedia/wikipedia`, 20231101) | CC BY-SA 3.0 + GFDL | ~58k paragraphs per language |
| Aozora Bunko, cleaned (`globis-university/aozorabunko-clean`) | CC BY 4.0 (texts are public domain) | ~58k paragraphs (Japanese, mostly pre-war literary style) |
| [Tatoeba](https://tatoeba.org) sentences | CC BY 2.0 FR — attribution: Tatoeba contributors | up to 40k utterances per language (1–3 sentences joined) |"""
LIC_NOWIKI = "Model weights and `punct_restore.py`: MIT. Trained on Aozora Bunko (public-domain texts, CC BY 4.0 compilation) and Tatoeba sentences (CC BY 2.0 FR, © Tatoeba contributors); please keep this attribution when redistributing."
LIC_WIKI = ("Model weights and `punct_restore.py`: MIT. **The training data includes Wikipedia text (CC BY-SA 3.0 / GFDL).** We regard trained weights as not being an adaptation of that text, "
            "but this is not settled law; if you need to be strictly safe, use the Wikipedia-free sibling model (`kanameaijp/punct-restore-multilingual-{size}`). "
            "Attribution: Wikipedia contributors (CC BY-SA 3.0 / GFDL), Aozora Bunko (CC BY 4.0 compilation, public-domain texts), Tatoeba contributors (CC BY 2.0 FR).")

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    for variant, base_dir in (("nowiki", here), ("wiki", os.path.join(here, "alt-wiki"))):
        for size in ("base", "small"):
            name, text = build(variant, size)
            p = os.path.join(base_dir, name); os.makedirs(p, exist_ok=True)
            open(os.path.join(p, "README.md"), "w").write(text)
            open(os.path.join(p, "punct_restore.py"), "w").write(open(os.path.join(here, "punct_restore.py")).read())
            print(p)
