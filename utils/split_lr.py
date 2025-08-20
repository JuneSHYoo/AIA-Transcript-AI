#!/usr/bin/env python3
# split_lr.py
import argparse, json, subprocess, sys
from pathlib import Path

def run(cmd):
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"[실패] {' '.join(cmd)} -> exit {e.returncode}")

def ffprobe_channels(src):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_streams", "-print_format", "json",
        str(src),
    ]
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    j = json.loads(p.stdout or "{}")
    s = (j.get("streams") or [{}])[0]
    return int(s.get("channels") or 0), str(s.get("channel_layout") or "")

def main():
    ap = argparse.ArgumentParser(description="Stereo 파일을 L/R mp3로 분리")
    ap.add_argument("input", help="입력 오디오 파일 (stereo 권장)")
    ap.add_argument("--out-dir", default=".", help="출력 폴더")
    ap.add_argument("--sr", type=int, default=8000, help="출력 sample rate (Hz)")
    ap.add_argument("--bitrate", default="16k", help="출력 비트레이트 (예: 16k, 32k)")
    ap.add_argument("--fail-on-mono", action="store_true", help="모노면 실패 처리")
    ap.add_argument("--force", action="store_true", help="기존 파일 덮어쓰기")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"[에러] 입력 파일을 찾을 수 없음: {src}")

    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    base = src.stem
    outL = outdir / f"{base}_L.mp3"
    outR = outdir / f"{base}_R.mp3"

    overwrite = ["-y"] if args.force else ["-n"]

    ch, layout = ffprobe_channels(src)
    if ch < 1:
        raise SystemExit(f"[에러] 오디오 스트림을 찾지 못함: {src}")
    if ch == 1 and args.fail_on_mono:
        raise SystemExit("[에러] 입력이 모노(1ch) 입니다. --fail-on-mono 때문에 중단합니다.")

    if ch == 1:
        # 모노면 좌/우 동일 복제
        print("[정보] 모노 입력 → 동일 내용으로 L/R 두 파일 생성")
        for out in (outL, outR):
            cmd = ["ffmpeg", *overwrite, "-i", str(src),
                   "-vn", "-ac", "1",
                   "-c:a", "libmp3lame", "-ar", str(args.sr), "-b:a", args.bitrate,
                   str(out)]
            run(cmd)
    else:
        # 스테레오 이상 → 앞의 두 채널(c0, c1)만 사용
        # pan=mono|c0 / pan=mono|c1 로 각각 분리 (ffmpeg 7.x 호환)
        # 왼쪽만 추출
        cmdL = ["ffmpeg", *overwrite, "-i", str(src),
                "-vn", "-af", "pan=1c|c0=c0",  # ← 수정: pan=1c|c0=c0
                "-ac", "1", "-c:a", "libmp3lame",
                "-ar", str(args.sr), "-b:a", args.bitrate,
                str(outL)]

        # 오른쪽만 추출
        cmdR = ["ffmpeg", *overwrite, "-i", str(src),
                "-vn", "-af", "pan=1c|c0=c1",  # ← 수정: pan=1c|c0=c1
                "-ac", "1", "-c:a", "libmp3lame",
                "-ar", str(args.sr), "-b:a", args.bitrate,
                str(outR)]

        run(cmdL); run(cmdR)

    print(f"[완료] {outL}")
    print(f"[완료] {outR}")

if __name__ == "__main__":
    main()
