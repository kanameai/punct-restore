#!/usr/bin/env python3
"""Generate the README figures (SVG, no dependencies).

Numbers are copied from the model cards (Evaluation -> Comparison with other open models,
first 300 paragraphs per test set). Examples are real outputs of
kanameaijp/punct-restore-multilingual-base (gpu/queue/022_eval_nowiki.sh demo).
Run:  python docs/make_figures.py   ->  docs/assets/*.svg
"""
import os, html
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)
FONT = "font-family=\"-apple-system,BlinkMacSystemFont,'Segoe UI','Hiragino Sans','Noto Sans CJK JP','Noto Sans JP','Microsoft YaHei','Malgun Gothic',sans-serif\""
C_BASE, C_SMALL, C_OG, C_PCS = "#2563eb", "#93c5fd", "#f59e0b", "#9ca3af"
INK, MUTED, GRID = "#111827", "#6b7280", "#e5e7eb"
esc = lambda s: html.escape(s, quote=True)

# --- wikipedia test sets (not in the training data of the published models) ---
LANGS = ["ja", "zh", "ko", "en", "de", "fr", "es", "it", "pt", "nl", "ru"]
WIKI = {  # base, small, oliverguhr, pcs_47lang
 "de": (.854, .818, .803, .646), "en": (.802, .746, .786, .583), "es": (.754, .714, None, .538),
 "fr": (.756, .730, .767, .530), "it": (.709, .675, .726, .473), "ja": (.784, .749, None, .511),
 "ko": (.817, .803, None, .720), "nl": (.844, .809, .786, .655), "pt": (.766, .726, None, .545),
 "ru": (.780, .735, None, .553), "zh": (.693, .662, None, .365)}
NAMES = ["punct-restore base (308M)", "punct-restore small (141M)",
         "oliverguhr multilang-large (559M)", "punctuators pcs_47lang"]
COLORS = [C_BASE, C_SMALL, C_OG, C_PCS]

def avg(idx, langs):
    v = [WIKI[l][idx] for l in langs]
    return sum(v) / len(v)

def legend(x, y, items, gap=22):
    s = []
    for i, (name, col) in enumerate(items):
        s.append(f'<rect x="{x}" y="{y + i*gap - 10}" width="12" height="12" rx="2" fill="{col}"/>'
                 f'<text x="{x+18}" y="{y + i*gap}" font-size="13" fill="{INK}">{esc(name)}</text>')
    return "".join(s)

def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" {FONT} role="img">'
            f'<title>{esc(title)}</title><rect width="{w}" height="{h}" rx="12" fill="#ffffff" stroke="{GRID}"/>{body}</svg>')

# 1) summary: averages
def summary():
    five = ["de", "en", "fr", "it", "nl"]
    panels = [("Wikipedia, 5 languages (de en fr it nl)", five, [0, 1, 2, 3]),
              ("Wikipedia, all 11 languages", LANGS, [0, 1, 3])]
    W, x0, barw = 960, 300, 560
    b = [f'<text x="24" y="38" font-size="20" font-weight="700" fill="{INK}">Punctuation F1 on text the model never saw in training</text>',
         f'<text x="24" y="62" font-size="13" fill="{MUTED}">micro-F1 over , . ? ! : (higher is better) · Wikipedia paragraphs, lower-cased, punctuation removed · first 300 paragraphs per language</text>']
    y = 96
    for title, langs, idxs in panels:
        b.append(f'<text x="24" y="{y}" font-size="15" font-weight="700" fill="{INK}">{esc(title)}</text>')
        y += 14
        for i in idxs:
            v = avg(i, langs)
            bw = barw * v
            b.append(f'<text x="{x0-10}" y="{y+16}" font-size="13" text-anchor="end" fill="{INK}">{esc(NAMES[i])}</text>'
                     f'<rect x="{x0}" y="{y+2}" width="{barw}" height="20" rx="3" fill="#f3f4f6"/>'
                     f'<rect x="{x0}" y="{y+2}" width="{bw:.1f}" height="20" rx="3" fill="{COLORS[i]}"/>'
                     f'<text x="{x0+bw+8:.1f}" y="{y+17}" font-size="13" font-weight="700" fill="{INK}">{v:.3f}</text>')
            y += 28
        y += 22
    b.append(f'<text x="24" y="{y}" font-size="12" fill="{MUTED}">oliverguhr supports only these 5 languages and was given cased input (an advantage for it).</text>'
             f'<text x="24" y="{y+18}" font-size="12" fill="{MUTED}">The test sets follow our own punctuation conventions. See "How to read the numbers" in the README.</text>')
    return svg(W, y + 40, "".join(b), "Average punctuation F1 on Wikipedia")

# 2) per language
def per_language():
    W, H = 960, 430
    left, top, ch = 60, 110, 240
    gw = (W - left - 30) / len(LANGS)
    bw = (gw - 16) / 4
    b = [f'<text x="24" y="38" font-size="20" font-weight="700" fill="{INK}">Per language · Wikipedia (out-of-domain for our models)</text>',
         f'<text x="24" y="62" font-size="13" fill="{MUTED}">punctuation micro-F1 · first 300 paragraphs per language · n/a = language not supported</text>']
    for i, n in enumerate(NAMES):
        col = i % 2; row = i // 2
        x = 24 + col * 330 + (row and 0)
        b.append(legend(24 + i * 230, 88, [(n, COLORS[i])]))
    for t in [0, .2, .4, .6, .8, 1.0]:
        yy = top + ch * (1 - t)
        b.append(f'<line x1="{left}" x2="{W-30}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{GRID}"/>'
                 f'<text x="{left-8}" y="{yy+4:.1f}" font-size="11" text-anchor="end" fill="{MUTED}">{t:.1f}</text>')
    for gi, l in enumerate(LANGS):
        gx = left + gi * gw + 8
        for i, v in enumerate(WIKI[l]):
            x = gx + i * bw
            if v is None:
                b.append(f'<text x="{x+bw/2:.1f}" y="{top+ch-4}" font-size="9" text-anchor="middle" fill="{MUTED}" transform="rotate(-90 {x+bw/2:.1f} {top+ch-4})">n/a</text>')
                continue
            h = ch * v
            b.append(f'<rect x="{x:.1f}" y="{top+ch-h:.1f}" width="{bw-2:.1f}" height="{h:.1f}" rx="2" fill="{COLORS[i]}"><title>{l} {NAMES[i]}: {v:.3f}</title></rect>')
        b.append(f'<text x="{gx + 2*bw:.1f}" y="{top+ch+20}" font-size="14" font-weight="700" text-anchor="middle" fill="{INK}">{l}</text>'
                 f'<text x="{gx + 2*bw:.1f}" y="{top+ch+38}" font-size="11" text-anchor="middle" fill="{C_BASE}">{WIKI[l][0]:.2f}</text>')
    b.append(f'<text x="24" y="{H-16}" font-size="12" fill="{MUTED}">Blue number = punct-restore base. Japanese: 0.78 vs 0.51 (pcs_47lang); Chinese: 0.69 vs 0.37; Korean: 0.82 vs 0.72.</text>')
    return svg(W, H, "".join(b), "Per-language punctuation F1 on Wikipedia")

# 3) before / after
EX = [("JA", "今日は朝から雨が降っていましたが午後には止んだので近所の公園まで散歩に出かけました",
              "今日は朝から雨が降っていましたが、午後には止んだので、近所の公園まで散歩に出かけました。"),
      ("JA", "すみません駅まではどう行けばいいですか", "すみません、駅まではどう行けばいいですか？"),
      ("EN", "what time does the train to berlin leave and do i need to change in hannover",
              "What time does the train to Berlin leave and do I need to change in Hannover?"),
      ("FR", "je ne sais pas si je peux venir demain mais je te préviens dès que possible",
              "Je ne sais pas si je peux venir demain, mais je te préviens dès que possible."),
      ("ZH", "我昨天去了超市买了一些水果和蔬菜然后回家做饭", "我昨天去了超市买了一些水果和蔬菜，然后回家做饭。"),
      ("KO", "오늘은 날씨가 좋아서 친구들과 함께 공원에 갔어요 정말 즐거웠어요", "오늘은 날씨가 좋아서 친구들과 함께 공원에 갔어요. 정말 즐거웠어요.")]

def marked(inp, out):
    """tspans: inserted punctuation in red, changed case in blue."""
    i, parts = 0, []
    for ch in out:
        if i < len(inp) and ch == inp[i]:
            parts.append(esc(ch)); i += 1
        elif i < len(inp) and ch.lower() == inp[i]:
            parts.append(f'<tspan fill="{C_BASE}" font-weight="700">{esc(ch)}</tspan>'); i += 1
        else:
            parts.append(f'<tspan fill="#dc2626" font-weight="800">{esc(ch)}</tspan>')
    return "".join(parts)

def hero():
    W = 960; row = 78; top = 92
    H = top + row * len(EX) + 40
    b = [f'<text x="24" y="40" font-size="22" font-weight="800" fill="{INK}">ASR output in, readable text out</text>',
         f'<text x="24" y="66" font-size="13" fill="{MUTED}">real outputs of punct-restore-multilingual-base · <tspan fill="#dc2626" font-weight="700">red</tspan> = punctuation added · <tspan fill="{C_BASE}" font-weight="700">blue</tspan> = capital letter restored</text>']
    for k, (lang, inp, out) in enumerate(EX):
        y = top + k * row
        b.append(f'<rect x="20" y="{y}" width="{W-40}" height="{row-10}" rx="8" fill="#f9fafb"/>'
                 f'<rect x="32" y="{y+12}" width="40" height="22" rx="11" fill="{INK}"/>'
                 f'<text x="52" y="{y+28}" font-size="12" font-weight="700" text-anchor="middle" fill="#fff">{lang}</text>'
                 f'<text x="88" y="{y+28}" font-size="14" fill="{MUTED}">{esc(inp)}</text>'
                 f'<text x="70" y="{y+54}" font-size="15" fill="{MUTED}">→</text>'
                 f'<text x="88" y="{y+54}" font-size="16" fill="{INK}">{marked(inp, out)}</text>')
    b.append(f'<text x="24" y="{H-16}" font-size="12" fill="{MUTED}">11 languages · one model · CPU: 10 short sentences in ~0.3 s</text>')
    return svg(W, H, "".join(b), "Before and after examples")

# 4) how it works
def how():
    W, H = 960, 200
    boxes = [("speech recognizer", "text without punctuation", "#f3f4f6"),
             ("token classifier", "mmBERT, 1 label per token", "#dbeafe"),
             ("12 labels", "{none , . ? ! :} × {lower, Upper}", "#dbeafe"),
             ("restore()", "renders 、。？！ / ，。 / ASCII", "#f3f4f6")]
    b = [f'<text x="24" y="36" font-size="18" font-weight="700" fill="{INK}">How it works</text>']
    bw, gap, y = 200, 36, 64
    for i, (t, s, col) in enumerate(boxes):
        x = 24 + i * (bw + gap)
        b.append(f'<rect x="{x}" y="{y}" width="{bw}" height="84" rx="10" fill="{col}" stroke="{GRID}"/>'
                 f'<text x="{x+bw/2}" y="{y+36}" font-size="15" font-weight="700" text-anchor="middle" fill="{INK}">{esc(t)}</text>'
                 f'<text x="{x+bw/2}" y="{y+60}" font-size="11.5" text-anchor="middle" fill="{MUTED}">{esc(s)}</text>')
        if i < len(boxes) - 1:
            ax = x + bw + 6
            b.append(f'<path d="M{ax} {y+42} h{gap-14} m-7 -6 l7 6 l-7 6" stroke="{MUTED}" stroke-width="2" fill="none"/>')
    b.append(f'<text x="24" y="{H-18}" font-size="12" fill="{MUTED}">Inputs longer than ~250 tokens are split into windows. Labels come from real punctuated text (no model-generated labels).</text>')
    return svg(W, H, "".join(b), "How it works")

for name, fn in [("hero", hero), ("benchmark_summary", summary), ("benchmark_per_language", per_language), ("how_it_works", how)]:
    with open(os.path.join(OUT, name + ".svg"), "w", encoding="utf-8") as f:
        f.write(fn())
    print("wrote", name)
print("avg5", [round(avg(i, ["de","en","fr","it","nl"]), 3) for i in range(4)])
print("avg11", [round(avg(i, LANGS), 3) for i in (0, 1, 3)])
