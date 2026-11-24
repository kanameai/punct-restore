"""publish/punct_restore.py の動作確認: 手書きの入力での出力、評価コードとの一致、速度、サイズ。usage: demo.py <model_dir> <out_md>"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import torch, common
from punct_restore import restore
from transformers import AutoModelForTokenClassification, AutoTokenizer
d, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(d); model = AutoModelForTokenClassification.from_pretrained(d).eval()
SENT = ["今日は朝から雨が降っていましたが午後には止んだので近所の公園まで散歩に出かけました",
 "すみません駅まではどう行けばいいですか",
 "会議は来週の火曜日に延期になりましたので資料の提出期限も合わせて変更してください",
 "我昨天去了超市买了一些水果和蔬菜然后回家做饭",
 "hello everyone welcome to the show today we are going to talk about how to train a small language model on a single gpu",
 "what time does the train to berlin leave and do i need to change in hannover",
 "wir haben uns gestern getroffen und dann sind wir zusammen ins kino gegangen",
 "je ne sais pas si je peux venir demain mais je te préviens dès que possible",
 "me gustaría reservar una mesa para cuatro personas el sábado por la noche",
 "오늘은 날씨가 좋아서 친구들과 함께 공원에 갔어요 정말 즐거웠어요"]
lines = []
for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
    model.to(dev); t0 = time.time(); outs = [restore(model, tok, s) for s in SENT]; dt = time.time() - t0
    lines.append(f"## device={dev}: {len(SENT)} 文 {dt:.2f}s（初回含む）")
    if dev == "cpu":
        for s, o in zip(SENT, outs): lines.append(f"- 入力: {s}\n  - 出力: {o}")
# 評価コードの予測と restore() が一致するか（句読点・大文字の位置）
model.to("cpu"); same = tot = 0
for t in common.read_texts("data/proc/test_tatoeba_en.jsonl", 40) + common.read_texts("data/proc/test_wikipedia_ja.jsonl", 40):
    e = common.encode(tok, t, 256)
    if not e: continue
    with torch.no_grad(): pred = model(input_ids=torch.tensor([e["input_ids"]])).logits[0].argmax(-1).tolist()
    pa, pu = common.decode(e["spans"], pred, len(e["inp"]))
    same += int(common.render(e["inp"], pa, pu).replace(" ", "") == restore(model, tok, e["inp"]).replace(" ", "")); tot += 1
lines.append(f"## restore() と評価コードの一致: {same}/{tot}")
lines.append(f"## パラメータ数: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M、model.safetensors: {os.path.getsize(d + '/model.safetensors') / 2**20:.0f} MiB")
open(out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
