#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import csv
import argparse
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple

# ------------------------------------
# Normalization
# ------------------------------------
def normalize_text(s: str, mode: str = "lenient") -> str:
    """
    NFC 정규화.
    - lenient: 소문자화 + 공백/구두점 제거(숫자/영문/한글/자모만 유지)
    - strict : 원문 그대로(단, NFC만)
    """
    s = unicodedata.normalize("NFC", s)
    if mode == "lenient":
        s = s.lower()
        s = re.sub(r"[^\w\u3130-\u318F\uAC00-\uD7AF]+", "", s, flags=re.UNICODE)
    return s

def normalize_with_map_mode(s: str, mode: str) -> Tuple[str, List[int], str]:
    """
    원문 s를 mode에 맞춰 정규화한 문자열과
    정규화문자 각 위치 → 원문 인덱스 매핑을 반환.
    returns: (norm_str, idx_map, raw_nfc)
    """
    raw_nfc = unicodedata.normalize("NFC", s)
    if mode == "strict":
        # strict는 문자 삭제가 없으므로 1:1 매핑
        norm = raw_nfc
        idx_map = list(range(len(raw_nfc)))
        return norm, idx_map, raw_nfc

    # lenient: 문자 삭제가 있으므로 다시 매핑을 구성
    norm_chars: List[str] = []
    idx_map: List[int] = []
    for i, ch in enumerate(raw_nfc):
        kept = normalize_text(ch, mode="lenient")
        if kept:
            # 보통 길이 1이지만 안전하게 반복
            for _ in kept:
                norm_chars.append(_)
                idx_map.append(i)
    return "".join(norm_chars), idx_map, raw_nfc

def slice_safe(s: str, a: int, b: int) -> str:
    if a < 0 or b < 0:
        return ""
    return s[a:b]

# ------------------------------------
# Levenshtein alignment (최적 편집경로)
# ------------------------------------
def levenshtein_alignment(a: List[str], b: List[str]):
    """
    문자(또는 토큰)열 a,b에 대해 최적 편집 경로를 복원해
    op 시퀀스('equal'/'replace'/'insert'/'delete', a0,a1,b0,b1)을 생성.
    """
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    bt = [[None]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = i; bt[i][0] = 1   # delete
    for j in range(1, m+1):
        dp[0][j] = j; bt[0][j] = 2   # insert
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            cost = 0 if ai == b[j-1] else 1
            diag = dp[i-1][j-1] + cost
            up   = dp[i-1][j] + 1
            left = dp[i][j-1] + 1
            if diag <= up and diag <= left:
                dp[i][j] = diag; bt[i][j] = 0
            elif up <= left:
                dp[i][j] = up;   bt[i][j] = 1
            else:
                dp[i][j] = left; bt[i][j] = 2

    # backtrace
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        move = bt[i][j]
        if move == 0:  # diag
            tag = "equal" if a[i-1] == b[j-1] else "replace"
            ops.append((tag, i-1, i, j-1, j))
            i -= 1; j -= 1
        elif move == 1:  # delete
            ops.append(("delete", i-1, i, j, j))
            i -= 1
        else:  # insert
            ops.append(("insert", i, i, j-1, j))
            j -= 1
    ops.reverse()

    # 인접 동일 op 병합
    merged = []
    for tag, a0, a1, b0, b1 in ops:
        if merged and merged[-1][0] == tag and a0 == merged[-1][2] and b0 == merged[-1][4]:
            mt, ma0, ma1, mb0, mb1 = merged.pop()
            merged.append((tag, ma0, a1, mb0, b1))
        else:
            merged.append((tag, a0, a1, b0, b1))
    return merged

# ------------------------------------
# difflib 기반 CSV (기존)
# ------------------------------------
def diff_csv_via_difflib(ans_path: str, stt_path: str, out_csv: str,
                         case_insensitive: bool = False, context: int = 8, mode: str = "lenient"):
    ans_norm, ans_map, ans_raw = normalize_with_map_mode(Path(ans_path).read_text(encoding="utf-8"), mode)
    stt_norm, stt_map, stt_raw = normalize_with_map_mode(Path(stt_path).read_text(encoding="utf-8"), mode)

    ans_cmp = ans_norm.lower() if case_insensitive else ans_norm
    stt_cmp = stt_norm.lower() if case_insensitive else stt_norm

    sm = SequenceMatcher(None, ans_cmp, stt_cmp, autojunk=False)
    opcodes = sm.get_opcodes()

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["diff_type", "ans_segment_raw", "stt_segment_raw", "ans_context_raw", "stt_context_raw"])
        for tag, a0, a1, b0, b1 in opcodes:
            if tag == "equal":
                continue
            def span_raw(idx_map, s0, s1):
                if s0 >= s1: return (-1, -1)
                return (idx_map[s0], idx_map[s1-1] + 1)

            ra0, ra1 = span_raw(ans_map, a0, a1)
            rb0, rb1 = span_raw(stt_map, b0, b1)

            segA = slice_safe(ans_raw, ra0, ra1)
            segB = slice_safe(stt_raw, rb0, rb1)
            ctxA = slice_safe(ans_raw, max(0, ra0 - context), ra1 + context) if ra0 >= 0 else ""
            ctxB = slice_safe(stt_raw, max(0, rb0 - context), rb1 + context) if rb0 >= 0 else ""

            w.writerow([tag, segA, segB, ctxA, ctxB])

# ------------------------------------
# Levenshtein 정렬 기반 CSV (정밀)
# ------------------------------------
def diff_csv_via_levenshtein(ans_path: str, stt_path: str, out_csv: str,
                             context: int = 8, mode: str = "lenient"):
    ans_norm, ans_map, ans_raw = normalize_with_map_mode(Path(ans_path).read_text(encoding="utf-8"), mode)
    stt_norm, stt_map, stt_raw = normalize_with_map_mode(Path(stt_path).read_text(encoding="utf-8"), mode)

    A = list(ans_norm)
    B = list(stt_norm)
    opcodes = levenshtein_alignment(A, B)

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["diff_type", "ans_segment_raw", "stt_segment_raw", "ans_context_raw", "stt_context_raw"])
        for tag, a0, a1, b0, b1 in opcodes:
            if tag == "equal":
                continue
            def span_raw(idx_map, s0, s1):
                if s0 >= s1: return (-1, -1)
                return (idx_map[s0], idx_map[s1-1] + 1)

            ra0, ra1 = span_raw(ans_map, a0, a1)
            rb0, rb1 = span_raw(stt_map, b0, b1)

            segA = slice_safe(ans_raw, ra0, ra1)
            segB = slice_safe(stt_raw, rb0, rb1)
            ctxA = slice_safe(ans_raw, max(0, ra0 - context), ra1 + context) if ra0 >= 0 else ""
            ctxB = slice_safe(stt_raw, max(0, rb0 - context), rb1 + context) if rb0 >= 0 else ""

            w.writerow([tag, segA, segB, ctxA, ctxB])

# ------------------------------------
# CLI
# ------------------------------------
def main():
    p = argparse.ArgumentParser(description="답안지 vs 전사결과 문자 단위 diff → CSV (Levenshtein 또는 difflib)")
    p.add_argument("answer_file", help="답안지 파일 경로")
    p.add_argument("stt_file", help="전사결과 파일 경로")
    p.add_argument("-o", "--out", default="diff_chunks.csv", help="출력 CSV 경로 (기본: diff_chunks.csv)")
    p.add_argument("--engine", choices=["levenshtein", "difflib"], default="levenshtein",
                   help="diff 엔진 선택 (기본: levenshtein)")
    p.add_argument("--mode", choices=["strict", "lenient"], default="lenient",
                   help="정규화 모드: strict(그대로) / lenient(공백·구두점 제거, 소문자)")
    p.add_argument("--ignore-case", action="store_true", help="(difflib 전용) 대소문자 무시")
    p.add_argument("--context", type=int, default=8, help="원문 컨텍스트 좌/우 글자 수 (기본 8)")
    args = p.parse_args()

    if args.engine == "levenshtein":
        diff_csv_via_levenshtein(
            ans_path=args.answer_file,
            stt_path=args.stt_file,
            out_csv=args.out,
            context=args.context,
            mode=args.mode,
        )
    else:
        diff_csv_via_difflib(
            ans_path=args.answer_file,
            stt_path=args.stt_file,
            out_csv=args.out,
            case_insensitive=args.ignore_case,
            context=args.context,
            mode=args.mode,
        )

    print(f"[완료] 차이 구간 CSV 저장: {args.out}")
    print(f"- 입력: {args.answer_file} vs {args.stt_file}")
    print(f"- 엔진: {args.engine} / 모드: {args.mode}")

if __name__ == "__main__":
    main()
