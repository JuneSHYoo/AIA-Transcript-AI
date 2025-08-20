#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import csv
import argparse
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple

ALLOW_RE = re.compile(r"[0-9A-Za-z가-힣]")

def normalize_with_map(s: str) -> Tuple[str, List[int]]:
    """
    원문 s에서 허용 문자(숫자/영문/한글)만 남기고 이어붙인 문자열을 반환.
    또한 정규화 문자열의 각 문자 인덱스 -> 원문 인덱스 매핑 리스트를 함께 반환.
      예) norm[i]는 raw[map[i]]에서 왔음.
    """
    norm_chars = []
    idx_map = []
    for i, ch in enumerate(s):
        if ALLOW_RE.match(ch):
            norm_chars.append(ch)
            idx_map.append(i)
    return "".join(norm_chars), idx_map

def span_raw_indices(idx_map: List[int], n0: int, n1: int) -> Tuple[int, int]:
    """
    정규화 문자열 구간 [n0, n1) 를 원문 인덱스 구간 [r0, r1) 로 근사 변환.
    공백 등 제거로 인해 비어있을 수 있으므로 예외 처리 포함.
    """
    if n0 >= n1:
        return (-1, -1)
    r0 = idx_map[n0]
    r1 = idx_map[n1 - 1] + 1  # 끝 다음 인덱스
    return (r0, r1)

def slice_safe(s: str, a: int, b: int) -> str:
    if a < 0 or b < 0:
        return ""
    return s[a:b]

def diff_chunks_to_csv(ans_path: str, stt_path: str, out_csv: str, case_insensitive: bool = False, context: int = 8):
    ans_raw = Path(ans_path).read_text(encoding="utf-8")
    stt_raw = Path(stt_path).read_text(encoding="utf-8")

    # 정규화 + 맵
    ans_norm, ans_map = normalize_with_map(ans_raw)
    stt_norm, stt_map = normalize_with_map(stt_raw)

    if case_insensitive:
        ans_cmp = ans_norm.lower()
        stt_cmp = stt_norm.lower()
    else:
        ans_cmp = ans_norm
        stt_cmp = stt_norm

    sm = SequenceMatcher(None, ans_cmp, stt_cmp)

    # headers = [
    #     "diff_type",
    #     "ans_norm_start","ans_norm_end","stt_norm_start","stt_norm_end",
    #     "ans_raw_start","ans_raw_end","stt_raw_start","stt_raw_end",
    #     "ans_segment_norm","stt_segment_norm",
    #     "ans_segment_raw","stt_segment_raw",
    #     "ans_context_raw","stt_context_raw",
    # ]
    headers = ["diff_type", "ans_segment_raw", "stt_segment_raw", "ans_context_raw", "stt_context_raw"]

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(headers)

        for tag, a0, a1, b0, b1 in sm.get_opcodes():
            if tag == "equal":
                continue

            # 정규화 구간을 원문 구간으로 매핑
            ra0, ra1 = span_raw_indices(ans_map, a0, a1)
            rb0, rb1 = span_raw_indices(stt_map, b0, b1)

            # 세그먼트 추출 (정규화/원문)
            seg_ans_norm = ans_norm[a0:a1]
            seg_stt_norm = stt_norm[b0:b1]
            seg_ans_raw  = slice_safe(ans_raw, ra0, ra1)
            seg_stt_raw  = slice_safe(stt_raw, rb0, rb1)

            # 원문 컨텍스트(좌우 각 context 글자)
            ans_ctx = slice_safe(ans_raw, max(0, ra0 - context), ra1 + context) if ra0 >= 0 else ""
            stt_ctx = slice_safe(stt_raw, max(0, rb0 - context), rb1 + context) if rb0 >= 0 else ""

            # w.writerow([
            #     tag,
            #     a0, a1, b0, b1,
            #     ra0, ra1, rb0, rb1,
            #     seg_ans_norm, seg_stt_norm,
            #     seg_ans_raw, seg_stt_raw,
            #     ans_ctx, stt_ctx,
            # ])
            w.writerow([tag, seg_ans_raw, seg_stt_raw, ans_ctx, stt_ctx])

    print(f"[완료] 차이 구간 CSV 저장: {out_csv}")
    print(f"- 비교 기준: 문자 단위 / 공백·구두점 무시 / {'대소문자 무시' if case_insensitive else '대소문자 구분'}")
    print(f"- 입력: {ans_path} vs {stt_path}")

def main():
    p = argparse.ArgumentParser(description="두 텍스트(답안지 vs 전사결과)를 문자 단위로 비교(공백/구두점 무시)하여 차이 구간만 CSV로 저장")
    p.add_argument("answer_file", help="답안지 파일 경로")
    p.add_argument("stt_file", help="전사결과 파일 경로")
    p.add_argument("-o", "--out", default="diff_chunks.csv", help="출력 CSV 경로 (기본: diff_chunks.csv)")
    p.add_argument("--ignore-case", action="store_true", help="대소문자 무시")
    p.add_argument("--context", type=int, default=8, help="원문 컨텍스트 좌/우 글자 수 (기본 8)")
    args = p.parse_args()

    diff_chunks_to_csv(
        ans_path=args.answer_file,
        stt_path=args.stt_file,
        out_csv=args.out,
        case_insensitive=args.ignore_case,
        context=args.context,
    )

if __name__ == "__main__":
    main()
