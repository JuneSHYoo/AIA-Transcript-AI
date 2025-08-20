# stt_file_cli.py
import os, json, subprocess, shlex, time
from pathlib import Path
from dotenv import load_dotenv
import argparse
from datetime import datetime, timezone # 시간 측정용
import azure.cognitiveservices.speech as speechsdk

# ===== 0) 환경 변수 =====
load_dotenv()
AZ_KEY = os.getenv("AZURE_SPEECH_KEY")
AZ_REGION = os.getenv("AZURE_SPEECH_REGION")
if not AZ_KEY or not AZ_REGION:
    raise ValueError(".env에 AZURE_SPEECH_KEY / AZURE_SPEECH_REGION 설정 필요")

# ===== 1) ffprobe로 오디오 스펙 확인 =====
def probe_audio(path: str) -> dict:
    cmd = f'ffprobe -v error -show_streams -select_streams a -of json {shlex.quote(path)}'
    out = subprocess.check_output(cmd, shell=True, text=True)
    info = json.loads(out)
    if not info.get("streams"):
        raise RuntimeError("오디오 스트림을 찾지 못했습니다.")
    s = info["streams"][0]
    return {
        "codec_name": s.get("codec_name"),
        "sample_rate": int(s.get("sample_rate", "0")),
        "channels": int(s.get("channels", 0)),
        "bits_per_sample": int(s.get("bits_per_sample", 0) or 0)
    }

# ===== 2) 필요하면 ffmpeg로 변환 (16k/mono/PCM16) =====
def ensure_azure_format(src_path: str) -> str:
    spec = probe_audio(src_path)
    need_convert = (
        spec["codec_name"] != "pcm_s16le" or
        spec["sample_rate"] != 16000 or
        spec["channels"] != 1 or
        spec["bits_per_sample"] != 16
    )

    convert_dir = Path("convert")
    convert_dir.mkdir(exist_ok=True)

    src = Path(src_path)
    dst = convert_dir / f"{src.stem}_16k.wav"

    if not need_convert:
        # 이미 권장 포맷이면 그대로 복사
        import shutil
        shutil.copy(src_path, dst)
        return str(dst)

    cmd = f'ffmpeg -y -i {shlex.quote(src_path)} -ac 1 -ar 16000 -sample_fmt s16 {shlex.quote(str(dst))}'
    print("🔁 변환 실행:", cmd)
    subprocess.check_call(cmd, shell=True)
    return str(dst)

# ===== 3) Azure STT (연속 인식) =====
def stt_file(path: str, lang="ko-KR") -> str:
    speech_config = speechsdk.SpeechConfig(subscription=AZ_KEY, region=AZ_REGION)
    speech_config.speech_recognition_language = lang

    audio_config = speechsdk.AudioConfig(filename=path)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    texts, done = [], False

    def on_recognized(evt):
        if evt.result.text:
            print("🔹", evt.result.text)
            texts.append(evt.result.text)

    def on_stop(_):
        nonlocal done
        done = True

    recognizer.recognized.connect(on_recognized)
    recognizer.session_stopped.connect(on_stop)
    recognizer.canceled.connect(on_stop)

    # ↓↓↓ STT 시간 측정 추가 ↓↓↓
    stt_start_iso = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    recognizer.start_continuous_recognition()
    while not done:
        time.sleep(0.2)
    recognizer.stop_continuous_recognition()

    t1 = time.perf_counter()
    stt_end_iso = datetime.now(timezone.utc).isoformat()
    stt_elapsed = t1 - t0
    # ↑↑↑ STT 시간 측정 추가 ↑↑↑
    return " ".join(texts), stt_start_iso, stt_end_iso, stt_elapsed

# ===== 4) CLI =====
def parse_args():
    p = argparse.ArgumentParser(description="Azure STT for local audio file(s)")
    p.add_argument("sources", nargs="*", help="파일 경로(복수 가능). 쉘 글롭 패턴도 가능: audio/*.mp3")
    p.add_argument("--lang", default="ko-KR", help="인식 언어 (기본: ko-KR)")
    p.add_argument("--outdir",default="stt_results", help="결과 저장 디렉토리 (기본: result/monostt)")

    return p.parse_args()

def main():
    args = parse_args()
    # 소스가 없으면 입력받기(한 줄)
    sources = args.sources
    if not sources:
        one = input("처리할 파일 경로를 입력하세요: ").strip()
        if not one:
            raise SystemExit("파일 경로가 없습니다.")
        sources = [one]

    # 쉘이 글롭을 확장 못하는 환경 대비: Path.glob 대체 지원
    paths: list[Path] = []
    for s in sources:
        p = Path(s)
        if any(ch in s for ch in "*?[]"):
            paths.extend(Path().glob(s))
        elif p.exists():
            paths.append(p)
        else:
            print(f"⚠ 경고: 경로를 찾을 수 없습니다: {s}")

    if not paths:
        raise SystemExit("처리할 파일이 없습니다.")

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    for src in paths:
        print(f"\n=== 파일 처리: {src} ===")
        try:
            norm_path = ensure_azure_format(str(src))
            print(f"✅ 사용 파일: {norm_path}")

            full_text, stt_start_iso, stt_end_iso, stt_elapsed = stt_file(norm_path, lang=args.lang)

            # 결과 저장
            out_txt = outdir / f"{Path(src).stem}.txt"
            out_txt.write_text(full_text, encoding="utf-8")
            print(f"💾 저장 완료: {out_txt}")

            # 시간 리포트
            print(f"⏱ STT 시작(UTC): {stt_start_iso}")
            print(f"⏱ STT 종료(UTC): {stt_end_iso}")
            print(f"⏱ STT 경과: {stt_elapsed:.3f}s")
        except subprocess.CalledProcessError as e:
            print(f"❌ ffmpeg/ffprobe 실행 실패: {e}")
        except Exception as e:
            print(f"❌ 오류: {e}")

if __name__ == "__main__":
    main()
