import os, json, subprocess, shlex, time
from pathlib import Path
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import csv
import argparse
from common.common_audio import inspect_and_prepare  #
from common.common_timing import Timer                 #
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

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
        "bits_per_sample": int(s.get("bits_per_sample", 0)),
    }

def is_azure_recommended(spec: dict) -> bool:
    return (
        spec["codec_name"] == "pcm_s16le" and
        spec["sample_rate"] == 16000 and
        spec["channels"] == 1 and
        spec["bits_per_sample"] == 16
    )

# ===== 2) 필요하면 ffmpeg로 변환 (16k/mono/PCM16) =====
def ensure_azure_format(src_path: str) -> str:
    """
    - 원본이 이미 권장 스펙이면 변환 없이 그대로 사용하지만,
      실행 경로 일관성을 위해 convert/<원본이름>_16k.wav 로 복사만 함.
    - 권장 스펙이 아니면 convert/ 에 변환 파일 생성.
    - 이미 만들어진 convert/ 파일이 있으면 재변환 없이 그대로 사용.
    """
    convert_dir = Path("convert")
    convert_dir.mkdir(exist_ok=True)

    src = Path(src_path)
    dst = convert_dir / f"{src.stem}_16k.wav"

    # convert/에 동일 산출물이 이미 있으면 그대로 사용
    if dst.exists():
        return str(dst)

    # 원본 스펙 확인
    spec = probe_audio(str(src))
    if is_azure_recommended(spec):
        # 권장 스펙이면 복사만 해서 convert/에서 쓰자
        import shutil
        shutil.copy(str(src), str(dst))
        return str(dst)

    # 변환 필요
    cmd = f'ffmpeg -y -i {shlex.quote(str(src))} -ac 1 -ar 16000 -sample_fmt s16 {shlex.quote(str(dst))}'
    print("🔁 Azure 권장 규격 변환 실행:", cmd)
    subprocess.check_call(cmd, shell=True)
    return str(dst)

# ===== 3) 공통: SpeechRecognizer 생성 헬퍼 =====
def create_recognizer(audio_file: str, lang="ko-KR") -> speechsdk.SpeechRecognizer:
    speech_config = speechsdk.SpeechConfig(subscription=AZ_KEY, region=AZ_REGION)
    speech_config.speech_recognition_language = lang
    audio_config = speechsdk.AudioConfig(filename=audio_file)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    return recognizer, speech_config

# ===== 4) 화자 분리 옵션(버전 호환형) 적용 =====
def enable_diarization(speech_config: speechsdk.SpeechConfig):
    """
    SDK 버전에 따라 속성명이 다른 경우가 있어, 가능한 후보들을 순차 시도.
    실패해도 예외 던지지 않고 그냥 넘어가도록 구성(스트리밍은 계속 동작).
    """
    candidates = [
        # 새/대체 키 후보들
        ("SpeechServiceConnection_DiarizationMode", "speaker"),
        ("SpeechServiceResponse_RequestDiarization", "true"),
        # 과거 문서/지역별 키 후보들
        ("SpeechServiceConnection_SpeakerDiarizationMode", "True"),
        ("SpeechServiceResponse_SpeakerDiarizationEnabled", "true"),
    ]
    for name, val in candidates:
        try:
            pid = getattr(speechsdk.PropertyId, name)
            speech_config.set_property(pid, val)
        except Exception:
            pass  # 지원 안 하면 조용히 스킵

    # (선택) 단어 타임스탬프
    try:
        speech_config.request_word_level_timestamps()
    except Exception:
        pass

# ===== 5) 연속 인식(기본) =====
def stt_file(path: str, lang="ko-KR"):
    recognizer, _ = create_recognizer(path, lang)
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

    recognizer.start_continuous_recognition()

    while not done:
        time.sleep(0.2)
    recognizer.stop_continuous_recognition()
    return " ".join(texts)

# ===== 6) 연속 인식 + 화자 분리 =====
def stt_file_with_diarization(path: str, lang="ko-KR"):
    recognizer, speech_config = create_recognizer(path, lang)
    enable_diarization(speech_config)

    segments, done = [], False

    def on_recognized(evt):
        res = evt.result
        if res.text:
            start_sec = res.offset / 10_000_000  # 100ns → s
            dur_sec = res.duration / 10_000_000
            speaker = getattr(res, "speaker_id", None)  # 지원되는 버전에서 채워짐
            seg = {
                "speaker": speaker,
                "start": round(start_sec, 2),
                "duration": round(dur_sec, 2),
                "text": res.text,
            }
            print(f"🔹 S{seg['speaker']} [{seg['start']}s ~ +{seg['duration']}s]: {seg['text']}")
            segments.append(seg)

    def on_stop(_):
        nonlocal done
        done = True

    recognizer.recognized.connect(on_recognized)
    recognizer.session_stopped.connect(on_stop)
    recognizer.canceled.connect(on_stop)

    recognizer.start_continuous_recognition()
    while not done:
        time.sleep(0.2)
    recognizer.stop_continuous_recognition()
    return segments


# ===== 6-1) 채널 라벨 기반 스트리밍 (L/R 분리 처리용) =====
def stt_file_channel_labeled(path: str, channel_label: str, lang="ko-KR"):
    recognizer, _ = create_recognizer(path, lang)
    segments, done = [], False

    def on_recognized(evt):
        res = evt.result
        if res.text:
            start_sec = res.offset / 10_000_000
            dur_sec = res.duration / 10_000_000
            segments.append({
                "speaker": None,
                "channel": channel_label,   # 'L' 또는 'R'
                "start": round(start_sec, 2),
                "duration": round(dur_sec, 2),
                "text": res.text,
            })
            print(f"🔹 {channel_label} [{round(start_sec,2)}s ~ +{round(dur_sec,2)}s]: {res.text}")

    def on_stop(_):
        nonlocal done
        done = True

    recognizer.recognized.connect(on_recognized)
    recognizer.session_stopped.connect(on_stop)
    recognizer.canceled.connect(on_stop)

    recognizer.start_continuous_recognition()
    while not done:
        time.sleep(0.2)
    recognizer.stop_continuous_recognition()
    return segments


# ===== 7) 저장 유틸 =====
def save_stream_results(base_path: Path, segments: list, formats: set):
    """
    base_path: stream_results/<stem>_stream (확장자 제외)
    segments: [{speaker, channel, start, duration, text}, ...]
    """
    base_path.parent.mkdir(parents=True, exist_ok=True)

    # TXT
    if "txt" in formats:
        lines = []
        for seg in segments:
            tag = f"S{seg['speaker']}" if seg.get("speaker") is not None else (seg.get("channel") or "?")
            lines.append(f"[{seg['start']:>7.2f}s] {tag}: {seg['text']}")
        (base_path.with_suffix(".txt")).write_text("\n".join(lines), encoding="utf-8")

    # CSV
    if "csv" in formats:
        with open(base_path.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["start_sec", "duration_sec", "speaker_id", "channel", "text"])
            for seg in segments:
                writer.writerow([seg["start"], seg["duration"], seg.get("speaker"), seg.get("channel"), seg["text"]])

    # JSON
    if "json" in formats:
        with open(base_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

# ===== 8) CLI =====
def parse_args():
    p = argparse.ArgumentParser(description="Azure STT Streaming (auto/mono/stereo)")
    p.add_argument("sources", nargs="*", help="파일 경로(복수 가능). 예: audio/*.mp3")
    p.add_argument("--lang", default="ko-KR", help="인식 언어 (기본: ko-KR)")
    p.add_argument("--outdir", default="stream_results", help="결과 저장 디렉토리 (기본: stream_results)")
    p.add_argument("--mode", choices=["auto", "mono", "stereo"], default="auto", help="처리 모드 (기본 auto)")
    p.add_argument("--formats", help="저장 형식 콤마구분: txt,csv,json (기본 3종 모두)")
    p.add_argument("--dialog", action="store_true", help="(스테레오) L/R 병합 대화록도 생성")
    return p.parse_args()


# ===== 실행 진입점 =====
if __name__ == "__main__":
    # [MODIFIED] 데모 → CLI 유틸로 변경
    args = parse_args()

    # 저장 형식 파싱
    if args.formats:
        formats = {s.strip().lower() for s in args.formats.split(",") if s.strip()}
        allowed = {"txt", "csv", "json"}
        formats = formats & allowed
        if not formats:
            print("⚠ 저장 형식이 유효하지 않아 기본 3종으로 진행합니다.")
            formats = {"txt", "csv", "json"}
    else:
        formats = {"txt", "csv", "json"}

    # 입력 수집 (글롭 보정)
    sources = args.sources or [input("처리할 파일 경로를 입력하세요: ").strip()]
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

        # 1) 공통 전처리 & 채널 분기 판정
        info = inspect_and_prepare(str(src))   # [ADDED] (.wma→.wav, mono/dual_mono/true_stereo 판별, 분기 준비)

        # 2) 시간 측정 공통 (모든 경우)
        with Timer():                          # [ADDED] 시작/종료/경과 자동 출력
            # 2-1) 모노/dual_mono → 모노 스트림
            if args.mode == "mono" or (args.mode == "auto" and info["kind"] in ("mono", "dual_mono")):
                norm = info["mono_wav"]
                segments = stt_file_with_diarization(norm, lang=args.lang)
                segments.sort(key=lambda x: x["start"])
                base = outdir / f"{Path(src).stem}_stream"   # {원본명}_stream.{txt,csv,json}
                save_stream_results(base, segments, formats)
                # Timer 가 여기서 자동으로 3줄 출력

            else:
                # 2-2) true_stereo → L/R 각각 스트림
                if info["kind"] != "true_stereo":
                    print("⚠ true_stereo가 아니어서 L/R 분리 이점이 없습니다. mono 처리 권장.")
                # Ls = stt_file_channel_labeled(info["L_wav"], "L", lang=args.lang)
                # Rs = stt_file_channel_labeled(info["R_wav"], "R", lang=args.lang)
                # Ls.sort(key=lambda x: x["start"]); Rs.sort(key=lambda x: x["start"])
                #
                # baseL = outdir / f"{Path(src).stem}_stream_L"   # {원본명}_stream_L.json (등)
                # baseR = outdir / f"{Path(src).stem}_stream_R"   # {원본명}_stream_R.json (등)
                # save_stream_results(baseL, Ls, formats)
                # save_stream_results(baseR, Rs, formats)

                def run_channel(label: str, wav_path: str):
                    # Todo:  필요 시 여기서 STT 클라이언트 생성(스레드-세이프가 아니면)
                    segs = stt_file_channel_labeled(wav_path, label, lang=args.lang)
                    segs.sort(key=lambda x: x["start"])
                    return label, segs


                with Timer():  # L/R 전체 처리 시간 측정
                    Ls = Rs = None
                    with ThreadPoolExecutor(max_workers=2) as ex:
                        futures = [
                            ex.submit(run_channel, "L", info["L_wav"]),
                            ex.submit(run_channel, "R", info["R_wav"]),
                        ]
                        for fut in as_completed(futures):
                            label, segs = fut.result()  # 예외 발생 시 여기서 즉시 raise
                            if label == "L":
                                Ls = segs
                            else:
                                Rs = segs

                baseL = outdir / f"{Path(src).stem}_stream_L"
                baseR = outdir / f"{Path(src).stem}_stream_R"
                save_stream_results(baseL, Ls, formats)
                save_stream_results(baseR, Rs, formats)

                # (옵션) 대화 병합
                if args.dialog and "json" in formats:
                    left_json  = str(baseL.with_suffix(".json"))
                    right_json = str(baseR.with_suffix(".json"))
                    dialog_prefix = outdir / f"{Path(src).stem}_stream_dialog"
                    try:
                        # cmd = f'python merge_dialog.py --left {shlex.quote(left_json)} --right {shlex.quote(right_json)} --label-left L --label-right R --out-prefix {shlex.quote(str(dialog_prefix))} --txt'
                        # subprocess.check_call(cmd, shell=True)
                        ## 경로 수정 후
                        cmd = [
                            sys.executable, "-m", "services.merge_dialog",
                            "--left", str(left_json),
                            "--right", str(right_json),
                            "--label-left", "L", "--label-right", "R",
                            "--out-prefix", str(dialog_prefix),
                            "--txt",
                            "--max-pause", "0.2",
                            "--phrase-split-seconds", "5",
                            "--fallback-window", "2",
                        ]
                        subprocess.check_call(cmd)  # shell=False
                    except Exception as e:
                        print("병합 실패(무시):", e)

    print("\n✅ 완료")