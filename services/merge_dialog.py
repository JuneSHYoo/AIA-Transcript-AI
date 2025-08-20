#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_dialog.py

두 개의 STT 결과(Left/Right, JSON 또는 CSV)를 시간축에 병합해
핑퐁 대화(turns)로 만들어 .json / .csv (+옵션: .txt)로 출력.

지원:
- JSON (Azure Batch/Whisper 계열):
  - 단어 레벨(words / displayWords) 타임스탬프가 있으면 우선 사용
  - 없으면 문장 레벨로 파싱 + 선택적 강제 분할(--phrase-split-seconds / --fallback-window)
  - 다양한 시간 표기(ISO8601 PT, ticks, ms, sec)를 자동 파싱
  - JSON5/주석/트레일링 콤마/words 배열의 흔한 형식오류를 관대하게 보정(load_json_file)
- CSV:
  - 컬럼: idx?, speaker?, start|start_sec, end|duration|duration_sec, text

출력:
- <out-prefix>.json  : [{start, end, speaker, text}, …]
- <out-prefix>.csv   : start,end,speaker,text
- <out-prefix>.txt   : [mm:ss.mmm-mm:ss.mmm] Speaker: text  (옵션 --txt)

예시:
  python merge_dialog.py \
    --left  batch_results/claim_L_min.json \
    --right batch_results/claim_R_min.json \
    --label-left Agent --label-right Customer \
    --out-prefix batch_results/claim_min_dialog \
    --max-pause 0.6 \
    --phrase-split-seconds 8 --fallback-window 3 \
    --txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TICKS_PER_SEC = 10_000_000  # Azure ticks (100ns)

# ----------------------------- tolerant JSON loader -----------------------------

def load_json_file(path: Path):
    """
    JSON/JSON5-ish/JSONL를 관대하게 읽는다.
    - BOM 제거
    - //, /* */ 주석 제거
    - 트레일링 콤마 제거
    - NaN/Infinity -> null
    - Azure words/displayWords 배열에서 흔한 '{{' or 누락 '{' 보정
    - 실패 시 JSON Lines 시도
    - 그래도 실패하면 주변 스니펫 포함 에러
    """
    s = path.read_text(encoding="utf-8", errors="ignore")
    s = s.lstrip("\ufeff")  # BOM 제거
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s2 = s
        # 주석 제거
        s2 = re.sub(r'(?m)^\s*//.*$', '', s2)
        s2 = re.sub(r'/\*.*?\*/', '', s2, flags=re.S)
        # 트레일링 콤마 제거
        s2 = re.sub(r',\s*([}\]])', r'\1', s2)
        # NaN/Infinity 정규화
        s2 = re.sub(r'\bNaN\b', 'null', s2)
        s2 = re.sub(r'\b-?Infinity\b', 'null', s2)

        # FIX1: words/displayWords 배열 시작 직후 {{ -> { 로 교정
        s2 = re.sub(r'("words"\s*:\s*\[\s*)\{+', r'\1{', s2, flags=re.I)
        s2 = re.sub(r'("displayWords"\s*:\s*\[\s*)\{+', r'\1{', s2, flags=re.I)

        # FIX2: 누락된 { 보정 (배열 첫 항목 & 중간 항목)
        s2 = re.sub(r'("words"\s*:\s*\[\s*)"word"\s*:', r'\1{"word":', s2, flags=re.I)
        s2 = re.sub(r'("displayWords"\s*:\s*\[\s*)"displayText"\s*:', r'\1{"displayText":', s2, flags=re.I)
        s2 = re.sub(r'}\s*,\s*"word"\s*:', r'}, {"word":', s2, flags=re.I)
        s2 = re.sub(r'}\s*,\s*"displayText"\s*:', r'}, {"displayText":', s2, flags=re.I)

        # 2차 시도
        try:
            return json.loads(s2)
        except json.JSONDecodeError as e:
            # JSON Lines 형태 시도
            try:
                lines = [json.loads(line) for line in s2.splitlines() if line.strip()]
                if lines:
                    return lines
            except Exception:
                pass
            # 친절한 에러
            ctx_start = max(e.pos - 120, 0)
            ctx_end = min(e.pos + 120, len(s2))
            snippet = s2[ctx_start:ctx_end]
            raise RuntimeError(
                f"JSON parse failed for {path} at char {e.pos} "
                f"(line {e.lineno}, col {e.colno}). Near:\n...{snippet}..."
            ) from e

# ----------------------------- helpers -----------------------------

def _num(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except Exception:
        return None

def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def _parse_pt_seconds(s: Any) -> Optional[float]:
    """'PT1H2M3.45S' 또는 'P1DT2H...' 형태 ISO8601 duration -> seconds(float)"""
    if not isinstance(s, str) or "P" not in s:
        return None
    m = re.fullmatch(
        r"P(?:(?P<d>\d+)D)?"
        r"(?:T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?)?",
        s
    )
    if not m:
        return None
    d = float(m.group("d") or 0)
    h = float(m.group("h") or 0)
    mi = float(m.group("m") or 0)
    se = float(m.group("s") or 0)
    return d * 86400 + h * 3600 + mi * 60 + se

def _time_from_fields(obj: Dict[str, Any]) -> Tuple[float, float]:
    """
    dict에서 범용적으로 start/end(초)를 계산.
    우선순위 (1) milliseconds 계열 → (2) ticks/ms/ISO8601 혼합 → (3) 기타 숫자 필드
    """
    # 1) milliseconds 계열 최우선
    ms_start = _num(obj.get("offsetMilliseconds"))
    ms_dur = _num(obj.get("durationMilliseconds"))
    ms_end = _num(obj.get("endTimeMilliseconds"))

    if ms_start is not None:
        s = ms_start / 1000.0
        if ms_end is not None:
            e = ms_end / 1000.0
        elif ms_dur is not None:
            e = s + (ms_dur / 1000.0)
        else:
            # duration도 end도 없으면 최소 길이 0.5초 가정(혹은 s 그대로)
            e = s
        return float(s), float(e)

    # 2) 그 외
    start = _first_not_none(
        (_num(obj.get("offsetInTicks")) / TICKS_PER_SEC) if _num(obj.get("offsetInTicks")) is not None else None,
        (_num(obj.get("offsetMilliseconds")) / 1000.0)   if _num(obj.get("offsetMilliseconds")) is not None else None,
        _parse_pt_seconds(obj.get("offset")),
        # 'Offset'이 ticks일 수도, sec일 수도 있어 heuristic
        (_num(obj.get("Offset")) / TICKS_PER_SEC) if (_num(obj.get("Offset")) is not None and _num(obj.get("Offset")) >= 2_000_000) else _num(obj.get("Offset")),
        _num(obj.get("StartTime")),
        _num(obj.get("startTime")),
        _num(obj.get("start")),
    )

    dur = _first_not_none(
        (_num(obj.get("durationInTicks")) / TICKS_PER_SEC) if _num(obj.get("durationInTicks")) is not None else None,
        (_num(obj.get("durationMilliseconds")) / 1000.0)   if _num(obj.get("durationMilliseconds")) is not None else None,
        _parse_pt_seconds(obj.get("duration")),
        (_num(obj.get("Duration")) / TICKS_PER_SEC) if (_num(obj.get("Duration")) is not None and _num(obj.get("Duration")) >= 2_000_000) else _num(obj.get("Duration")),
    )

    endv = _first_not_none(
        (_num(obj.get("endTimeInTicks")) / TICKS_PER_SEC) if _num(obj.get("endTimeInTicks")) is not None else None,
        (_num(obj.get("endTimeMilliseconds")) / 1000.0)   if _num(obj.get("endTimeMilliseconds")) is not None else None,
        _parse_pt_seconds(obj.get("endTime")),
        _parse_pt_seconds(obj.get("EndTime")),
        _num(obj.get("end")),
        _num(obj.get("End")),
    )

    if start is None:
        start = 0.0
    if endv is None:
        endv = start + (dur if dur is not None else 0.5)
    return float(start), float(endv)

def _to_list_candidates(data: Any):
    """
    Azure 결과에서 문장/세그먼트 배열을 찾아 리턴.
    시간 정보가 있는 recognizedPhrases/results/phrases를 **우선** 사용.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # 1) 시간정보가 확실한 것들 먼저
        for k in ("recognizedPhrases", "results", "phrases", "Phrases"):
            v = data.get(k)
            if isinstance(v, list) and v:
                return v

        # 2) AudioFileResults 내부 구조 (다수 파일 포함 케이스)
        afr = data.get("AudioFileResults")
        if isinstance(afr, list):
            out = []
            for af in afr:
                for k in ("RecognizedPhrases", "CombinedRecognizedPhrases", "SegmentResults"):
                    v = af.get(k)
                    if isinstance(v, list):
                        out.extend(v)
            if out:
                return out

        # 3) 마지막 수단: combinedRecognizedPhrases (보통 타임스탬프 없음)
        v = data.get("combinedRecognizedPhrases")
        if isinstance(v, list) and v:
            return v

        return [data]
    return []

# ----------------------------- core types -----------------------------

@dataclass
class Atom:
    start: float
    end: float
    speaker: str
    text: str

# ----------------------------- parsers -----------------------------

def parse_json_atoms(path: Path, default_speaker: str,
                     phrase_split_seconds: float = 0.0,
                     fallback_window: float = 0.0) -> List[Atom]:
    """
    JSON -> Atom[]
    - 미니멀 배열형: [{start|start_sec, end|duration|duration_sec, text, speaker?}, ...]
    - Azure Batch 형: recognizedPhrases / words / displayWords 등
    """
    data = load_json_file(path)

    # ---- 0) 미니멀 배열형 빠른 처리 ----
    if isinstance(data, list) and data and isinstance(data[0], dict):
        atoms: List[Atom] = []
        for it in data:
            s = _first_not_none(_num(it.get("start")), _num(it.get("start_sec")))
            e = _num(it.get("end"))
            d = _first_not_none(_num(it.get("duration")), _num(it.get("duration_sec")))
            txt = (it.get("text") or "").strip()
            spk = it.get("speaker") or default_speaker
            if s is None or not txt:
                continue
            if e is None:
                e = s + float(d or 0.0)
            atoms.append(Atom(float(s), float(e), spk, txt))
        atoms.sort(key=lambda x: (x.start, x.end))
        if atoms:
            return atoms

    # ---- 1) Azure-ish ----
    phrases = _to_list_candidates(data)
    out: List[Atom] = []

    for ph in phrases:
        # 상단 가설
        nbest = ph.get("NBest") or ph.get("nBest") or ph.get("nbest")
        hyp0 = nbest[0] if isinstance(nbest, list) and nbest else None

        # 단어 레벨
        words = None
        if hyp0:
            words = hyp0.get("Words") or hyp0.get("words") or hyp0.get("displayWords")
        if words is None:
            words = ph.get("Words") or ph.get("words") or ph.get("displayWords")

        # 문장 레벨 시간
        start_phrase, end_phrase = _time_from_fields(ph)

        if isinstance(words, list) and words:
            seg_tokens = []
            seg_start = None
            last_end = None

            def flush():
                nonlocal seg_tokens, seg_start, last_end
                if seg_tokens:
                    seg_text = " ".join(seg_tokens).strip()
                    out.append(Atom(seg_start, last_end or seg_start, default_speaker, seg_text))
                    seg_tokens, seg_start, last_end = [], None, None

            for w in words:
                token = (w.get("word") or w.get("Word") or w.get("displayText") or w.get("text"))
                if not token:
                    continue
                token = str(token).strip()
                ws, we = _time_from_fields(w)

                if seg_start is None:
                    seg_start, last_end, seg_tokens = ws, we, [token]
                    continue

                # 분할 조건
                cur_len = (last_end - seg_start) if (last_end is not None and seg_start is not None) else 0.0
                # 1) 긴 구간이면 시간 기준 분할
                exceed_len = (phrase_split_seconds and cur_len >= phrase_split_seconds)
                # 2) 단어 사이 간격이 크면 분할 (fallback_window를 gap 기준으로 사용)
                gap = ws - (last_end or ws)
                long_gap = (fallback_window and gap > fallback_window)
                # 3) 구두점(문장 끝) + 일정 길이 이상이면 분할
                hard_punct = bool(re.search(r"[.!?…]+$", token))
                punct_split = hard_punct and phrase_split_seconds and (cur_len >= min(phrase_split_seconds * 0.5, 4.0))

                if exceed_len or long_gap or punct_split:
                    flush()
                    seg_start, last_end, seg_tokens = ws, we, [token]
                else:
                    seg_tokens.append(token)
                    last_end = we

            flush()
            continue

        # 문장 레벨 텍스트 (display > lexical > itn > maskedITN > text)
        text = _first_not_none(
            hyp0.get("display") if hyp0 else None,
            ph.get("display"),
            hyp0.get("lexical") if hyp0 else None,
            ph.get("lexical"),
            hyp0.get("itn") if hyp0 else None,
            ph.get("itn"),
            hyp0.get("maskedITN") if hyp0 else None,
            ph.get("maskedITN"),
            ph.get("text"),
        )
        if not text:
            continue
        text = re.sub(r"\s+", " ", str(text)).strip()

        # 너무 긴 문장 강제 분할 옵션
        phrase_len = max(0.0, end_phrase - start_phrase)
        segs: List[tuple] = []
        need_split = phrase_split_seconds and phrase_len > phrase_split_seconds
        if need_split:
            # 1차 분할: 문장부호 기준
            parts: List[str] = []
            last = 0
            for m in re.finditer(r'([\.!?…]+)', text):
                endi = m.end()
                seg = text[last:endi].strip()
                if seg:
                    parts.append(seg)
                last = endi
            if last < len(text):
                tail = text[last:].strip()
                if tail:
                    parts.append(tail)

            if parts:
                total_chars = sum(len(p) for p in parts) or len(parts)
                cur = start_phrase
                for p in parts:
                    share = (len(p) / total_chars) if total_chars else (1.0 / len(parts))
                    d = phrase_len * share
                    segs.append((cur, min(cur + d, end_phrase), p))
                    cur += d

            # 여전히 길면 고정 윈도우로 토막내기
            if fallback_window and any((b - a) > phrase_split_seconds for a, b, _ in segs):
                refined = []
                for a, b, ptext in segs:
                    durp = b - a
                    if durp <= phrase_split_seconds:
                        refined.append((a, b, ptext))
                        continue
                    n = max(1, int(durp // fallback_window))
                    tokens = ptext.split() or [ptext]
                    words_per = max(1, len(tokens) // n)
                    t0 = a
                    i = 0
                    while i < len(tokens):
                        chunk = tokens[i:i + words_per]; i += words_per
                        t1 = min(b, t0 + fallback_window)
                        refined.append((t0, t1, " ".join(chunk)))
                        t0 = t1
                    if t0 < b:
                        refined[-1] = (refined[-1][0], b, refined[-1][2])
                segs = refined

        if not segs:
            segs = [(start_phrase, end_phrase, text)]

        for a, b, t in segs:
            out.append(Atom(a, b, default_speaker, t))

    out.sort(key=lambda x: (x.start, x.end))
    return out

def parse_csv_atoms(path: Path, default_speaker: str) -> List[Atom]:
    out: List[Atom] = []
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                start = _first_not_none(_num(row.get("start_sec")), _num(row.get("start")))
                end = _num(row.get("end"))
                dur = _first_not_none(_num(row.get("duration_sec")), _num(row.get("duration")))
                txt = (row.get("text") or "").strip()
                if start is None or not txt:
                    continue
                if end is None:
                    end = start + float(dur or 0.0)
                out.append(Atom(float(start), float(end), default_speaker, txt))
            except Exception:
                continue
    out.sort(key=lambda x: (x.start, x.end))
    return out

# ----------------------------- grouping & saving -----------------------------

def group_atoms_to_turns(atoms: List[Atom], max_pause: float = 0.6) -> List[Dict[str, Any]]:
    # atoms = sorted(atoms, key=lambda a: (a.start, a.speaker))
    atoms = sorted(atoms, key=lambda a: (a.start, a.end))
    turns: List[Dict[str, Any]] = []

    cur: Optional[Dict[str, Any]] = None
    for a in atoms:
        if cur is None:
            cur = {"speaker": a.speaker, "start": a.start, "end": a.end, "text": a.text}
            continue
        if a.speaker == cur["speaker"] and (a.start - cur["end"]) <= max_pause:
            cur["text"] = (cur["text"] + " " + a.text).strip()
            cur["end"] = max(cur["end"], a.end)
        else:
            turns.append(cur)
            cur = {"speaker": a.speaker, "start": a.start, "end": a.end, "text": a.text}
    if cur:
        turns.append(cur)
    return turns

def save_json(turns: List[Dict[str, Any]], path: Path):
    path.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")

def save_csv(turns: List[Dict[str, Any]], path: Path):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "speaker", "text"])
        for t in turns:
            w.writerow([round(t["start"], 3), round(t["end"], 3), t["speaker"], t["text"]])

def _fmt_ts(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:06.3f}"  # mm:ss.mmm

def save_txt(turns: List[Dict[str, Any]], path: Path, with_time: bool = True):
    lines = []
    for t in turns:
        spk = t["speaker"]
        txt = t["text"]
        if with_time:
            lines.append(f"[{_fmt_ts(t['start'])}-{_fmt_ts(t['end'])}] {spk}: {txt}")
        else:
            lines.append(f"{spk}: {txt}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ----------------------------- cli -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Merge two STT results (JSON/CSV) into a dialog")
    ap.add_argument("--left",  required=True, help="Left/CH0 file (.json or .csv)")
    ap.add_argument("--right", required=False, help="Right/CH1 file (.json or .csv)")
    ap.add_argument("--label-left",  default="A")
    ap.add_argument("--label-right", default="B")
    ap.add_argument("--out-prefix", required=True, help="출력 파일 접두(확장자 제외)")
    ap.add_argument("--max-pause", type=float, default=0.6, help="동일 화자 발화 병합 허용 최대 간격(초)")
    ap.add_argument("--phrase-split-seconds", type=float, default=0.0, help="문장 레벨 강제 분할 임계값(초). 0=비활성")
    ap.add_argument("--fallback-window", type=float, default=0.0, help="강제 분할 후에도 길면 이 윈도우(초)로 추가 분할")
    ap.add_argument("--txt", action="store_true", help=".txt 대화록도 함께 저장")
    ap.add_argument("--txt-no-timestamps", action="store_true", help=".txt에 타임스탬프 생략")
    args = ap.parse_args()

    left_path = Path(args.left)
    right_path = Path(args.right) if args.right else None

    atoms: List[Atom] = []

    # Left
    if left_path.suffix.lower() == ".json":
        atoms += parse_json_atoms(left_path, args.label_left, args.phrase_split_seconds, args.fallback_window)
    elif left_path.suffix.lower() == ".csv":
        atoms += parse_csv_atoms(left_path, args.label_left)
    else:
        print(f"Unsupported left file type: {left_path.suffix}", file=sys.stderr)
        sys.exit(2)

    # Right (optional)
    if right_path:
        if right_path.suffix.lower() == ".json":
            atoms += parse_json_atoms(right_path, args.label_right, args.phrase_split_seconds, args.fallback_window)
        elif right_path.suffix.lower() == ".csv":
            atoms += parse_csv_atoms(right_path, args.label_right)
        else:
            print(f"Unsupported right file type: {right_path.suffix}", file=sys.stderr)
            sys.exit(2)

    if not atoms:
        print("No content parsed.", file=sys.stderr)
        sys.exit(3)

    turns = group_atoms_to_turns(atoms, max_pause=args.max_pause)
    # turns.sort(key=lambda x: (x["start"], x["speaker"]))
    turns.sort(key=lambda x: (x["start"]))
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    save_json(turns, out_prefix.with_suffix(".json"))
    save_csv(turns, out_prefix.with_suffix(".csv"))
    if args.txt:
        save_txt(turns, out_prefix.with_suffix(".txt"), with_time=not args.txt_no_timestamps)

    outputs = {
        "json": str(out_prefix.with_suffix(".json")),
        "csv":  str(out_prefix.with_suffix(".csv")),
    }
    if args.txt:
        outputs["txt"] = str(out_prefix.with_suffix(".txt"))

    print(json.dumps({
        "turns": len(turns),
        "outputs": outputs
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
