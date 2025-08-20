
import unicodedata
import argparse
import re
from typing import Tuple, List

def normalize_text(s: str, mode: str = "strict") -> str:
    """
    NFC normalize. If mode == 'lenient', remove spaces and punctuation,
    fold multiple whitespace to single, and lowercase latin.
    """
    s = unicodedata.normalize("NFC", s)
    if mode == "lenient":
        # Lowercase latin/numbers unaffected
        s = s.lower()
        # Remove punctuation and spaces
        # Keep Korean syllables, Hangul Jamo (if any), letters, numbers
        s = re.sub(r"[^\w\u3130-\u318F\uAC00-\uD7AF]+", "", s, flags=re.UNICODE)
    return s

def levenshtein_ops(a: List[str], b: List[str]) -> Tuple[int, int, int, int]:
    """
    Wagner-Fischer with backtrace to count insertions, deletions, substitutions.
    Returns (distance, ins, del_, sub)
    """
    n, m = len(a), len(b)
    # dp and backpointers: 0 diag(match/sub), 1 up(del), 2 left(ins)
    dp = [[0]*(m+1) for _ in range(n+1)]
    bt = [[None]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = i
        bt[i][0] = 1
    for j in range(1, m+1):
        dp[0][j] = j
        bt[0][j] = 2
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            choices = [
                (dp[i-1][j-1] + cost, 0),  # diag
                (dp[i-1][j] + 1, 1),       # up (del)
                (dp[i][j-1] + 1, 2),       # left (ins)
            ]
            best, move = min(choices, key=lambda x: x[0])
            dp[i][j] = best
            bt[i][j] = move
    # backtrace
    i, j = n, m
    ins = del_ = sub = 0
    while i > 0 or j > 0:
        move = bt[i][j]
        if move == 0:
            # diag: match or sub
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

def compute_metrics(ref: str, hyp: str, mode: str = "strict") -> dict:
    ref_n = normalize_text(ref, mode)
    hyp_n = normalize_text(hyp, mode)
    A = list(ref_n)
    B = list(hyp_n)
    lenA, lenB = len(A), len(B)

    dist, ins, delt, sub = levenshtein_ops(A, B)
    # symmetric similarity
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

def main():
    p = argparse.ArgumentParser(description="Compute syllable-based similarity between REF and HYP texts.")
    p.add_argument("--ans", type=str, required=True, help="답안지 text file path")
    p.add_argument("--stt", type=str, required=True, help="Azure STT text file path (e.g., Azure STT result)")
    p.add_argument("--mode", choices=["strict","lenient","both"], default="both",
                   help="Normalization mode: strict (raw), lenient (spaces/punct removed), both")
    args = p.parse_args()

    with open(args.ref, "r", encoding="utf-8") as f:
        ref = f.read()
    with open(args.hyp, "r", encoding="utf-8") as f:
        hyp = f.read()

    modes = ["strict","lenient"] if args.mode == "both" else [args.mode]
    for m in modes:
        res = compute_metrics(ref, hyp, m)
        print(f"\n=== Mode: {m} ===")
        for k, v in res.items():
            if k in ("edit_similarity","A_based_match_ratio","B_based_match_ratio","lcs_similarity"):
                print(f"{k:>24}: {v:.4f}")
            else:
                print(f"{k:>24}: {v}")

if __name__ == "__main__":
    main()
