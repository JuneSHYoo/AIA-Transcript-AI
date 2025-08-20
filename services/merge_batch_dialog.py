#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_LR_dialog.py

두 개의 STT 결과(Left/Right, JSON 또는 CSV)를 시간축에 병합해
핑퐁 대화(turns)로 .json / .csv (+옵션: .txt)로 출력.

모드:
- pause (기본): 동일 화자 + max_pause 이하 gap이면 병합
- slice_offset: 시작시각(anchor) 기준 offset_window 초를 넘으면 새 턴
- slice_duration: 한 턴의 최대 길이가 max_duration 초를 넘지 않도록 분할

word 분할(기본값 활성):
- --word-gap (기본 0.45s)
- --word-max-len (기본 3.0s)
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
    """JSON/JSON5-ish/JSONL를 관대하게 읽음."""
    s = path.read_text(encoding="utf-8", errors="ignore").lstrip("\ufeff")  # BOM 제거
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
        # 흔한 배열 형식 오류 보정
        s2 = re.sub(r'("words"\s*:\s*\[\s*)\{+', r'\1{', s2, flags=re.I)
        s2 = re.sub(r'("displayWords"\s*:\s*\[\s*)\{+', r'\1{', s2, flags=re.I)
        s2 = re.sub(r'("words"\s*:\s*\[\s*)"word"\s*:', r'\1{"word":', s2, flags=re.I)
        s2 = re.sub(r'("displayWords"\s*:\s*\[\s*)"displayText"\s*:', r'\1{"displayText":', s2, flags=re.I)
        s2 = re.sub(r'}\s*,\s*"word"\s*:', r'}, {"word":', s2, flags=re.I)
        s2 = re.sub(r'}\s*,\s*"displayText"\s*:', r'}, {"displayText":', s2, flags=re.I)
        try:
            return json.loads(s2)
        except json.JSONDecodeError as e:
            # JSON Lines fallback
            try:
                lines = [json.loads(line) for line in s2.splitlines() if line.strip()]
                if lines:
                    return lines
            except Exception:
                pass
            ctx_start = max(e.pos - 120, 0)
            ctx_end = min(e.pos + 120, len(s2))
            snippet = s2[ctx_start:ctx_end]
            raise RuntimeError(
                f"JSON parse failed for {path} at char {e.pos} "
                f"(line {e.lineno}, col {e.colno}). Near:\n...{snippet}..."
            ) from e

# ----------------------------- helpers -----------------------------

def _get_ci(d: dict, key: str):
    """dict d에서 key를 대소문자 무시하고 찾아 값(Truthy) 반환."""
    if not isinstance(d, dict):
        return None
    # 흔히 쓰는 변형들 순회
    for kk in (key, key.lower(), key.upper(), key.capitalize()):
        v = d.get(kk)
        if v:
            return v
    # 일부 공급자가 snake/camel 섞는 경우 대비
    alt = key.replace("_", "").lower()
    for k in d.keys():
        if isinstance(k, str) and k.replace("_", "").lower() == alt and d.get(k):
            return d[k]
    return None


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
    """'PT1H2M3.45S' / 'P1DT2H...' ISO8601 duration -> seconds(float)"""
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
    우선순위: (1) *Milliseconds → (2) ticks/ISO → (3) 기타 숫자
    """
    # (1) ms 계열 최우선
    ms_start = _num(obj.get("offsetMilliseconds"))
    ms_dur   = _num(obj.get("durationMilliseconds"))
    ms_end   = _num(obj.get("endTimeMilliseconds"))
    if ms_start is not None:
        s = ms_start / 1000.0
        if ms_end is not None:
            e = ms_end / 1000.0
        elif ms_dur is not None:
            e = s + (ms_dur / 1000.0)
        else:
            e = s
        return float(s), float(e)

    # (2) 기타 시간 표기
    start = _first_not_none(
        (_num(obj.get("offsetInTicks")) / TICKS_PER_SEC) if _num(obj.get("offsetInTicks")) is not None else None,
        _parse_pt_seconds(obj.get("offset")),
        (_num(obj.get("Offset")) / TICKS_PER_SEC) if (_num(obj.get("Offset")) is not None and _num(obj.get("Offset")) >= 2_000_000) else _num(obj.get("Offset")),
        _num(obj.get("StartTime")),
        _num(obj.get("startTime")),
        _num(obj.get("start")),
    )
    dur = _first_not_none(
        (_num(obj.get("durationInTicks")) / TICKS_PER_SEC) if _num(obj.get("durationInTicks")) is not None else None,
        _parse_pt_seconds(obj.get("duration")),
        (_num(obj.get("Duration")) / TICKS_PER_SEC) if (_num(obj.get("Duration")) is not None and _num(obj.get("Duration")) >= 2_000_000) else _num(obj.get("Duration")),
    )
    endv = _first_not_none(
        (_num(obj.get("endTimeInTicks")) / TICKS_PER_SEC) if _num(obj.get("endTimeInTicks")) is not None else None,
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
    """Azure 결과에서 문장/세그먼트 배열을 찾아 리턴."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("recognizedPhrases", "results", "phrases", "Phrases"):
            v = data.get(k)
            if isinstance(v, list) and v:
                return v
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
def parse_json_atoms(
    path: Path,
    default_speaker: str,
    phrase_split_seconds: float = 0.0,
    fallback_window: float = 0.0,
    word_gap: Optional[float] = None,
    word_max_len: Optional[float] = None,
    text_field: str = "auto",  # ← NEW: 선택 필드
) -> List[Atom]:
    """
    JSON -> Atom[]
    - 단어 레벨(words/displayWords)이 있으면 우선 사용(기본 분할 활성)
    - 없으면 문장 레벨로 파싱 + 선택 분할
    - text_field: auto | display | lexical | itn | maskedITN | text
    """
    data = load_json_file(path)

    # 미니멀 배열형 바로 처리
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

    phrases = _to_list_candidates(data)
    out: List[Atom] = []

    # 안전 기본값: 옵션 미지정 시에도 분할되도록
    eff_gap = 0.45 if (word_gap is None and (fallback_window in (0, None))) else (word_gap if word_gap is not None else fallback_window or 0.45)
    eff_len = 3.0 if (word_max_len is None and (phrase_split_seconds in (0, None))) else (word_max_len if word_max_len is not None else phrase_split_seconds or 3.0)

    # 선택 가능한 텍스트 키 맵 (hyp0/phrase 모두에서 시도)
    FIELD_KEYS = {
        "display":   ("display",),
        "lexical":   ("lexical",),
        "itn":       ("itn",),
        "maskedITN": ("maskedITN", "maskedItn", "masked_itn"),
        "text":      ("text",),
    }
    text_field = (text_field or "auto").lower()

    for ph in phrases:
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

        # 단어 레벨 분할 우선
        if isinstance(words, list) and words:
            seg_tokens: List[str] = []
            seg_start: Optional[float] = None
            last_end: Optional[float] = None

            def flush():
                nonlocal seg_tokens, seg_start, last_end
                if seg_tokens:
                    seg_text = " ".join(seg_tokens).strip()
                    out.append(Atom(seg_start, last_end or seg_start, default_speaker, seg_text))
                    seg_tokens, seg_start, last_end = [], None, None

            for w in words:
                # 토큰 텍스트 선택 (옵션에 따라 itn/lexical/displayText/word/text 우선)
                def pick_token_text(w: dict) -> Optional[str]:
                    if text_field == "itn":
                        return _get_ci(w, "itn") or _get_ci(w, "ITN") \
                            or _get_ci(w, "displayText") or _get_ci(w, "word") or _get_ci(w, "text")
                    if text_field == "lexical":
                        # 토큰에 lexical이 없을 수 있어 displayText/word/text로 폴백
                        return _get_ci(w, "lexical") or _get_ci(w, "displayText") or _get_ci(w, "word") or _get_ci(w,
                                                                                                                   "text")
                    if text_field == "display":
                        return _get_ci(w, "displayText") or _get_ci(w, "word") or _get_ci(w, "text")
                    # auto/기타: 기존 순서 유지(가급적 보기 좋은 displayText 우선)
                    return _get_ci(w, "displayText") or _get_ci(w, "word") or _get_ci(w, "text") or _get_ci(w,
                                                                                                            "itn") or _get_ci(
                        w, "ITN")

                token = pick_token_text(w)
                if not token:
                    continue
                token = str(token).strip()
                ws, we = _time_from_fields(w)

                if seg_start is None:
                    seg_start, last_end, seg_tokens = ws, we, [token]
                    continue

                cur_len = (last_end - seg_start) if (last_end is not None and seg_start is not None) else 0.0
                gap = ws - (last_end or ws)
                hard_punct = bool(re.search(r"[.!?…]+$", token))

                exceed_len = (eff_len and cur_len >= eff_len)
                long_gap   = (eff_gap and gap > eff_gap)
                punct_split = hard_punct and (cur_len >= min(eff_len * 0.5, 2.0))

                if exceed_len or long_gap or punct_split:
                    flush()
                    seg_start, last_end, seg_tokens = ws, we, [token]
                else:
                    seg_tokens.append(token)
                    last_end = we

            flush()
            continue

        # 문장 레벨 텍스트 선택
        def pick_text_by_field(field: str) -> Optional[str]:
            # hyp0 우선, 없으면 phrase에서 케이스 무시로 탐색
            if field == "display":
                return _get_ci(hyp0 or {}, "display") or _get_ci(ph, "display")
            if field == "lexical":
                return _get_ci(hyp0 or {}, "lexical") or _get_ci(ph, "lexical")
            if field == "itn":
                return _get_ci(hyp0 or {}, "itn") or _get_ci(hyp0 or {}, "ITN") or _get_ci(ph, "itn") or _get_ci(ph, "ITN")
            if field == "maskedITN":
                # 다양한 철자/대소문자 대응
                for k in ("maskedITN", "MaskedITN", "maskedItn", "masked_itn"):
                    v = _get_ci(hyp0 or {}, k) or _get_ci(ph, k)
                    if v:
                        return v
                return None
            if field == "text":
                return _get_ci(ph, "text")
            return None

        if text_field != "auto":
            field_key = text_field  # already lower()
            if field_key == "maskeditn":
                field_key = "maskedITN"
            text = pick_text_by_field(field_key)
        else:
            # 자동 우선순위: display → lexical → itn → maskedITN → text (대소문자 변형 허용)
            text = (
                pick_text_by_field("display")
                or pick_text_by_field("lexical")
                or pick_text_by_field("itn")
                or pick_text_by_field("maskedITN")
                or pick_text_by_field("text")
            )


        if not text:
            continue
        text = re.sub(r"\s+", " ", str(text)).strip()

        phrase_len = max(0.0, end_phrase - start_phrase)
        segs: List[tuple] = []
        need_split = eff_len and phrase_len > eff_len
        if need_split:
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

            if eff_gap and any((b - a) > eff_len for a, b, _ in segs):
                refined = []
                for a, b, ptext in segs:
                    durp = b - a
                    if durp <= eff_len:
                        refined.append((a, b, ptext))
                        continue
                    n = max(1, int(durp // eff_gap))
                    tokens = ptext.split() or [ptext]
                    words_per = max(1, len(tokens) // n)
                    t0 = a
                    i = 0
                    while i < len(tokens):
                        chunk = tokens[i:i + words_per]; i += words_per
                        t1 = min(b, t0 + eff_gap)
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

# ----------------------------- grouping (merge) -----------------------------

def group_atoms_to_turns(atoms: List[Atom], max_pause: float = 0.6) -> List[Dict[str, Any]]:
    """
    모드: pause
    - 동일 화자이고, 이전 end와 다음 start의 차이가 max_pause 이하이면 병합
    """
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


def group_atoms_by_offset_window(
    atoms: List[Atom],
    offset_window: float,
    forbid_cross_speaker: bool = True
) -> List[Dict[str, Any]]:
    """
    모드: slice_offset
    - 현재 turn의 anchor(첫 토큰 start)에서 offset_window를 넘기면 새 turn
    - 화자 바뀌면 무조건 새 turn
    """
    atoms = sorted(atoms, key=lambda a: (a.start, a.end))
    turns: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    anchor: Optional[float] = None
    for a in atoms:
        if cur is None:
            cur = {"speaker": a.speaker, "start": a.start, "end": a.end, "text": a.text}
            anchor = a.start
            continue
        cross_speaker = (a.speaker != cur["speaker"])
        exceed = (a.start - (anchor if anchor is not None else a.start)) > offset_window
        if (forbid_cross_speaker and cross_speaker) or exceed:
            turns.append(cur)
            cur = {"speaker": a.speaker, "start": a.start, "end": a.end, "text": a.text}
            anchor = a.start
        else:
            cur["text"] = (cur["text"] + " " + a.text).strip()
            cur["end"] = max(cur["end"], a.end)
    if cur:
        turns.append(cur)
    return turns


def group_atoms_by_max_duration(
    atoms: List[Atom],
    max_duration: float,
    forbid_cross_speaker: bool = True
) -> List[Dict[str, Any]]:
    """
    모드: slice_duration
    - 현재 turn 시작(anchor) 기준으로 (현재 a.end - anchor) > max_duration 이면 새 turn
    - 화자 바뀌면 무조건 새 turn
    """
    atoms = sorted(atoms, key=lambda a: (a.start, a.end))
    turns: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    anchor: Optional[float] = None
    for a in atoms:
        if cur is None:
            cur = {"speaker": a.speaker, "start": a.start, "end": a.end, "text": a.text}
            anchor = a.start
            continue
        cross_speaker = (a.speaker != cur["speaker"])
        exceed = (a.end - (anchor if anchor is not None else a.start)) > max_duration
        if (forbid_cross_speaker and cross_speaker) or exceed:
            turns.append(cur)
            cur = {"speaker": a.speaker, "start": a.start, "end": a.end, "text": a.text}
            anchor = a.start
        else:
            cur["text"] = (cur["text"] + " " + a.text).strip()
            cur["end"] = max(cur["end"], a.end)
    if cur:
        turns.append(cur)
    return turns

# ----------------------------- saving -----------------------------

def save_json(turns: List[Dict[str, Any]], path: Path):
    path.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")

def save_csv(turns: List[Dict[str, Any]], path: Path, ndigits: int = 3):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "speaker", "text"])
        for t in turns:
            w.writerow([round(t["start"], ndigits), round(t["end"], ndigits), t["speaker"], t["text"]])

def _fmt_ts(sec: float, ndigits: int = 3) -> str:
    m = int(sec // 60)
    s = round(sec - m * 60, ndigits)
    frac_width = max(0, ndigits)
    # mm:ss.xxx
    return f"{m:02d}:{s:0{3+1+frac_width}.{ndigits}f}"

def save_txt(turns: List[Dict[str, Any]], path: Path, with_time: bool = True, ndigits: int = 3):
    lines = []
    for t in turns:
        spk = t["speaker"]
        txt = t["text"]
        if with_time:
            lines.append(f"[{_fmt_ts(t['start'], ndigits)}-{_fmt_ts(t['end'], ndigits)}] {spk}: {txt}")
        else:
            lines.append(f"{spk}: {txt}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ----------------------------- speaker switching -----------------------------
def _count_speaker_switches(turns: List[Dict[str, Any]]) -> int:
    """
    인접 턴 사이 화자 전환 횟수.
    예) R,R,L,L,R -> 전환 2회(R->L, L->R)
    """
    if not turns:
        return 0
    last = turns[0]["speaker"]
    sw = 0
    for t in turns[1:]:
        if t["speaker"] != last:
            sw += 1
            last = t["speaker"]
    return sw

# ----------------------------- cli -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Merge two STT results (JSON/CSV) into a dialog")
    ap.add_argument("--left",  required=True, help="Left/CH0 file (.json or .csv)")
    ap.add_argument("--right", required=False, help="Right/CH1 file (.json or .csv)")
    ap.add_argument("--label-left",  default="A")
    ap.add_argument("--label-right", default="B")
    # ✅ 기본 출력 디렉토리 추가
    ap.add_argument("--outdir", default="dialog_out", help="기본 출력 디렉토리(상대/절대 경로 모두 가능)")
    ap.add_argument("--out-prefix", required=True, help="출력 파일 접두(확장자 제외). 상대 경로면 --outdir 아래에 저장")

    # 분할/머지 파라미터
    ap.add_argument("--mode", choices=["pause", "slice_offset", "slice_duration"], default="pause",
                    help="병합 모드: pause | slice_offset | slice_duration")
    ap.add_argument("--max-pause", type=float, default=0.6, help="pause 모드: 동일 화자 병합 허용 최대 gap(초)")
    ap.add_argument("--offset-window", type=float, default=1.5, help="slice_offset 모드: anchor 기준 창(초)")
    ap.add_argument("--max-duration", type=float, default=3.0, help="slice_duration 모드: 턴 최대 길이(초)")

    # word 분할(옵션 미지정 시 기본값 활성)
    ap.add_argument("--word-gap", type=float, default=None, help="단어 사이 gap 분할 임계(초), 기본 0.45")
    ap.add_argument("--word-max-len", type=float, default=None, help="한 세그 최대 길이(초), 기본 3.0")

    # 문장 레벨 보조 분할(옵션)
    ap.add_argument("--phrase-split-seconds", type=float, default=0.0, help="문장 레벨 강제 분할 임계(초)")
    ap.add_argument("--fallback-window", type=float, default=0.0, help="추가 분할 윈도우(초)")

    # 출력
    ap.add_argument("--round-sec", type=int, default=3, help="CSV/TXT 출력 시 반올림 자리수")
    ap.add_argument("--txt", action="store_true", help=".txt 대화록도 함께 저장")
    ap.add_argument("--txt-no-timestamps", action="store_true", help=".txt에 타임스탬프 생략")

    # ✅ 화자 전환 없으면 한 줄 폴백
    ap.add_argument("--single-line-if-noswitch", action="store_true",
                    help="화자 전환이 0회면 모든 텍스트를 한 줄로 합치기")
    # 텍스트 필드 선택 (auto 또는 특정 필드 강제)
    ap.add_argument(
        "--text-field",
        choices=["auto", "display", "lexical", "itn", "maskedITN", "text"],
        default="auto",
        help="STT JSON에서 사용할 텍스트 필드 선택. 기본 auto(우선순위: display→lexical→itn→maskedITN→text)"
    )

    args = ap.parse_args()

    left_path = Path(args.left)
    right_path = Path(args.right) if args.right else None

    atoms: List[Atom] = []

    # Left
    if left_path.suffix.lower() == ".json":
        atoms += parse_json_atoms(
            left_path, args.label_left,
            phrase_split_seconds=args.phrase_split_seconds,
            fallback_window=args.fallback_window,
            word_gap=args.word_gap, word_max_len=args.word_max_len,
            text_field=args.text_field,  # ← NEW
        )
    elif left_path.suffix.lower() == ".csv":
        atoms += parse_csv_atoms(left_path, args.label_left)
    else:
        print(f"Unsupported left file type: {left_path.suffix}", file=sys.stderr)
        sys.exit(2)

    # Right (optional)
    if right_path:
        if right_path.suffix.lower() == ".json":
            atoms += parse_json_atoms(
                right_path, args.label_right,
                phrase_split_seconds=args.phrase_split_seconds,
                fallback_window=args.fallback_window,
                word_gap=args.word_gap, word_max_len=args.word_max_len,
                text_field=args.text_field,  # ← NEW
            )
        elif right_path.suffix.lower() == ".csv":
            atoms += parse_csv_atoms(right_path, args.label_right)
        else:
            print(f"Unsupported right file type: {right_path.suffix}", file=sys.stderr)
            sys.exit(2)

    if not atoms:
        print("No content parsed.", file=sys.stderr)
        sys.exit(3)

    # 병합 모드
    if args.mode == "pause":
        turns = group_atoms_to_turns(atoms, max_pause=args.max_pause)
    elif args.mode == "slice_offset":
        turns = group_atoms_by_offset_window(atoms, offset_window=args.offset_window, forbid_cross_speaker=True)
    else:  # slice_duration
        turns = group_atoms_by_max_duration(atoms, max_duration=args.max_duration, forbid_cross_speaker=True)

    # 정렬 + 저장
    turns.sort(key=lambda x: x["start"])
    # ✅ 화자 전환이 없으면 한 줄로 폴백
    if args.single_line_if_noswitch and _count_speaker_switches(turns) == 0 and len(turns) > 1:
        merged_text = " ".join(t["text"] for t in turns).strip()
        turns = [{
            "speaker": turns[0]["speaker"],
            "start": turns[0]["start"],
            "end": turns[-1]["end"],
            "text": merged_text
        }]

    # ✅ outdir 처리: out-prefix가 상대면 outdir 밑으로 저장
    raw_prefix = Path(args.out_prefix)
    out_prefix = raw_prefix if raw_prefix.is_absolute() else (Path(args.outdir) / raw_prefix)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    save_json(turns, out_prefix.with_suffix(".json"))
    save_csv(turns, out_prefix.with_suffix(".csv"), ndigits=args.round_sec)
    if args.txt:
        save_txt(turns, out_prefix.with_suffix(".txt"), with_time=not args.txt_no_timestamps, ndigits=args.round_sec)

    print(json.dumps({
        "turns": len(turns),
        "outputs": {
            "json": str(out_prefix.with_suffix(".json")),
            "csv": str(out_prefix.with_suffix(".csv")),
            **({"txt": str(out_prefix.with_suffix(".txt"))} if args.txt else {})
        }
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
