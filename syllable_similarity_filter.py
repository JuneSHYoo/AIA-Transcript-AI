#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unicodedata
import argparse
import re
from typing import Tuple, List

_PUNCT_WS = r"\s\.,!?~·…\"'()\[\]{}:;:/\\-"
def remove_fillers(text: str, fillers: List[str]) -> str:
    """
    공백/구두점 경계에서만 추임새를 제거.
    예: '네', '음' 등 단독 발화나 단어 사이에 있는 경우만 삭제하고,
        단어 내부에 섞인 경우는 보존.
    """
    if not fillers:
        return text
    # 길이 긴 것부터 매칭(중복·부분 일치 방지)
    fillers = sorted(set(fillers), key=len, reverse=True)
    alt = "|".join(map(re.escape, fillers))
    pattern = rf"(?:(?<=^)|(?<=[{_PUNCT_WS}]))(?:{alt})(?=(?:$|[{_PUNCT_WS}]))"
    text = re.sub(pattern, "", text)
    # 다중 공백 정리
    text = re.sub(r"\s{2,}", " ", text)
    return text

# --------------------
# Normalizers / Tokenizers
# --------------------
def normalize_text(s: str, mode: str = "strict") -> str:
    """
    NFC normalize. If mode == 'lenient', remove spaces and punctuation,
    lowercase latin. Used for character/음절-level comparison.
    """
    s = unicodedata.normalize("NFC", s)
    if mode == "lenient":
        s = s.lower()
        # Remove everything except letters/numbers/Hangul syllables & jamo
        s = re.sub(r"[^\w\u3130-\u318F\uAC00-\uD7AF]+", "", s, flags=re.UNICODE)
    return s

def word_tokenize(s: str, mode: str = "strict") -> List[str]:
    """
    Tokenize for WER. Keep spaces as boundaries.
    - strict: NFC, collapse whitespace; strip leading/trailing spaces. Keep words incl. numbers & Hangul.
    - lenient: NFC + lowercase + remove punctuation; collapse whitespace.
    """
    s = unicodedata.normalize("NFC", s)
    if mode == "lenient":
        s = s.lower()
        # Remove punctuation but keep whitespace so words remain separable
        s = re.sub(r"[^\w\s\u3130-\u318F\uAC00-\uD7AF]+", " ", s, flags=re.UNICODE)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    return s.split(" ")

# --------------------
# Edit distance utilities
# --------------------
def levenshtein_ops(a: List[str], b: List[str]) -> Tuple[int, int, int, int]:
    """
    Wagner-Fischer with backtrace to count insertions, deletions, substitutions.
    Returns (distance, ins, del_, sub)
    """
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    bt = [[None]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = i
        bt[i][0] = 1  # up (deletion)
    for j in range(1, m+1):
        dp[0][j] = j
        bt[0][j] = 2  # left (insertion)
    for i in range(1, n+1):
        ai = a[i-1]
        row = dp[i]
        prev_row = dp[i-1]
        bt_row = bt[i]
        for j in range(1, m+1):
            cost = 0 if ai == b[j-1] else 1
            # diag, up, left
            diag = prev_row[j-1] + cost
            up   = prev_row[j] + 1
            left = row[j-1] + 1
            if diag <= up and diag <= left:
                row[j] = diag
                bt_row[j] = 0  # diag
            elif up <= left:
                row[j] = up
                bt_row[j] = 1  # up (del)
            else:
                row[j] = left
                bt_row[j] = 2  # left (ins)

    # backtrace to count ops
    i, j = n, m
    ins = del_ = sub = 0
    while i > 0 or j > 0:
        move = bt[i][j]
        if move == 0:
            if a[i-1] != b[j-1]:
                sub += 1
            i -= 1; j -= 1
        elif move == 1:
            del_ += 1
            i -= 1
        elif move == 2:
            ins += 1
            j -= 1
        else:
            break
    return dp[n][m], ins, del_, sub

def lcs_length(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        ai = a[i-1]
        row = dp[i]
        prev_row = dp[i-1]
        for j in range(1, m+1):
            if ai == b[j-1]:
                row[j] = prev_row[j-1] + 1
            else:
                row[j] = row[j-1] if row[j-1] >= prev_row[j] else prev_row[j]
    return dp[n][m]

# --------------------
# Metrics
# --------------------
def compute_metrics_char(ref: str, hyp: str, mode: str = "strict") -> dict:
    ref_n = normalize_text(ref, mode)
    hyp_n = normalize_text(hyp, mode)
    A = list(ref_n)
    B = list(hyp_n)
    lenA, lenB = len(A), len(B)

    dist, ins, delt, sub = levenshtein_ops(A, B)
    sim_edit = 1.0 - (dist / max(1, (lenA + lenB)))
    matches = (lenA + lenB - dist) / 2.0
    prec_A = matches / max(1, lenA)
    rec_B  = matches / max(1, lenB)

    L = lcs_length(A, B)
    sim_lcs = (2.0 * L) / max(1, (lenA + lenB))

    return {
        "mode": mode,
        "len_ref": lenA,
        "len_hyp": lenB,
        "lev_distance": dist,
        "insertions": ins,
        "deletions": delt,
        "substitutions": sub,
        "edit_similarity": sim_edit,   # 1 - d/(|A|+|B|)
        "matches_estimate": matches,
        "A_based_match_ratio": prec_A, # matches/|A|
        "B_based_match_ratio": rec_B,  # matches/|B|
        "lcs_length": L,
        "lcs_similarity": sim_lcs      # 2L/(|A|+|B|)
    }

def compute_wer(ref: str, hyp: str, mode: str = "strict") -> float:
    """
    Compute word error rate using word-level Levenshtein.
    WER = (S + D + I) / N_words_in_ref
    """
    ref_tokens = word_tokenize(ref, mode)
    hyp_tokens = word_tokenize(hyp, mode)
    dist, ins, delt, sub = levenshtein_ops(ref_tokens, hyp_tokens)
    N = max(1, len(ref_tokens))
    return (ins + delt + sub) / N

# --------------------
# Pretty print (Korean)
# --------------------
def print_metrics_korean(res: dict, cer: float, char_accuracy: float, wer: float | None):
    mode_desc = "공백,구두점 포함" if res['mode'] == "strict" else "공백,구두점 제거"
    print(f"=== Mode: {res['mode']} ===")
    print(f"모드                 : {res['mode']} ({mode_desc})")
    print(f"답안지 길이          : {res['len_ref']}")
    print(f"STT 결과 길이        : {res['len_hyp']}")
    print(f"치환 오류(잘못 인식) : {res['substitutions']}")
    print(f"편집기반 일치율      : {res['edit_similarity']:.4f}")
    print(f"답안지 기준 일치율   : {res['A_based_match_ratio']:.4f}")
    print(f"STT 기준 일치율      : {res['B_based_match_ratio']:.4f}")
    print(f"LCS 기반 일치율      : {res['lcs_similarity']:.4f}")
    print(f"CER (음절 일치율)    : {char_accuracy:.4f}")
    print(f"CER (음절 오류율)    : {cer:.4f}")
    if wer is not None:
        print(f"WER (단어 오류율)    : {wer:.4f}")
    else:
        print(f"WER (단어 오류율)    : -")
    print()

# --------------------
# CLI
# --------------------
def main():
    p = argparse.ArgumentParser(description="Compare REF(정답) vs HYP(STT) at syllable & word level with Korean summary.")
    p.add_argument("--ans", type=str, required=True, help="답안지 text file path")
    p.add_argument("--stt", type=str, required=True, help="Azure STT text file path")
    p.add_argument("--mode", choices=["strict","lenient","both"], default="both",
                   help="Normalization mode for character-level: strict(raw) / lenient(spaces/punct removed) / both")
    # ▼ 추가 옵션
    p.add_argument("--ignore-fillers", action="store_true", help="추임새(네/예/음/어/아 등) 제거 후 계산")
    p.add_argument("--fillers", nargs="*", default=None,
                   help="제거할 추임새 목록을 공백 구분으로 지정 (미지정 시 기본 세트 사용)")
    args = p.parse_args()

    with open(args.ans, "r", encoding="utf-8") as f:
        ref = f.read()
    with open(args.stt, "r", encoding="utf-8") as f:
        hyp = f.read()

    # ▼ 추임새 제거(원문 단계에서 적용 → strict/lenient 모두에 일관됨)
    if args.ignore_fillers:
        fillers = args.fillers
        ref = remove_fillers(ref, fillers)
        hyp = remove_fillers(hyp, fillers)

    modes = ["strict","lenient"] if args.mode == "both" else [args.mode]
    for m in modes:
        # Character-level metrics (음절 기준)
        res = compute_metrics_char(ref, hyp, m)
        cer = (res["substitutions"] + res["deletions"] + res["insertions"]) / max(1, res["len_ref"])
        char_accuracy = 1 - cer  # 음절 일치율
        # Word-level WER (단어 기준)
        wer = compute_wer(ref, hyp, m)
        print_metrics_korean(res, cer, char_accuracy, wer)

if __name__ == "__main__":
    main()
