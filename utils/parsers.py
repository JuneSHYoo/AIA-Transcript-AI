import csv
import json
import re
from pathlib import Path

###################################
## JSON → 행 리스트 파싱 + CSV 저장  ##
###################################
def _to_seconds(val):
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val) if val < 1e6 else float(val) / 10_000_000.0
    s = str(val).strip()
    if s.isdigit():
        v = int(s)
        return float(v) if v < 1e6 else v / 10_000_000.0
    if s.startswith("PT"):
        h = m = 0.0
        sec = 0.0
        mh = re.search(r'(\d+(?:\.\d+)?)H', s)
        mm = re.search(r'(\d+(?:\.\d+)?)M', s)
        ms = re.search(r'(\d+(?:\.\d+)?)S', s)
        if mh: h = float(mh.group(1))
        if mm: m = float(mm.group(1))
        if ms: sec = float(ms.group(1))
        return h*3600 + m*60 + sec
    m = re.match(r'(\d+):([0-5]?\d):([0-5]?\d(?:\.\d+)?)', s)
    if m:
        hh = int(m.group(1)); mm = int(m.group(2)); ss = float(m.group(3))
        return hh*3600 + mm*60 + ss
    return 0.0

def parse_transcription_json(json_path: Path, stereo_mode=True, channel_labels=None):
    """
    stereo_mode=True  : speaker 없으면 channel 사용(CH0/CH1). channel_labels로 라벨링 가능
    stereo_mode=False : diarization 결과(speaker) 사용
    """
    j = json.loads(json_path.read_text(encoding="utf-8"))
    phrases = j.get("recognizedPhrases") or j.get("combinedRecognizedPhrases") or []

    rows = []
    for ph in phrases:
        if "offsetMilliseconds" in ph and "durationMilliseconds" in ph:
            start = round(float(ph["offsetMilliseconds"]) / 1000.0, 2)
            duration = round(float(ph["durationMilliseconds"]) / 1000.0, 2)
        else:
            start = round(_to_seconds(ph.get("offset", 0)), 2)
            duration = round(_to_seconds(ph.get("duration", 0)), 2)

        text = ""
        nbest = ph.get("nBest") or []
        if nbest:
            text = nbest[0].get("display") or nbest[0].get("lexical") or ""
        else:
            text = ph.get("display") or ph.get("lexical") or ph.get("text") or ""

        if stereo_mode:
            ch = ph.get("channel")
            if channel_labels and isinstance(ch, int) and ch in channel_labels:
                spk = channel_labels[ch]
            else:
                spk = f"CH{ch}" if ch is not None else "?"
        else:
            spk = ph.get("speaker")
            if spk is None:
                # diarization 없는 경우 안전장치
                ch = ph.get("channel")
                spk = f"CH{ch}" if ch is not None else "?"

        rows.append({"speaker": spk, "start": start, "duration": duration, "text": text})

    rows.sort(key=lambda r: (r["start"], str(r["speaker"])))
    return rows

def save_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "speaker", "start_sec", "duration_sec", "text"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["speaker"], r["start"], r["duration"], r["text"]])
