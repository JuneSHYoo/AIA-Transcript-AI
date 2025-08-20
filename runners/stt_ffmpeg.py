import os, json, subprocess, shlex, time
from pathlib import Path
from dotenv import load_dotenv
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
        "bits_per_sample": int(s.get("bits_per_sample", 0))
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

    # convert 폴더 생성
    convert_dir = Path("../convert")
    convert_dir.mkdir(exist_ok=True)

    # 변환 후 저장 경로
    src = Path(src_path)
    dst = convert_dir / f"{src.stem}_16k.wav"

    if not need_convert:
        # 이미 권장 포맷이면 그대로 convert 폴더에 복사
        import shutil
        shutil.copy(src_path, dst)
        return str(dst)

    # 변환 실행
    cmd = f'ffmpeg -y -i {shlex.quote(src_path)} -ac 1 -ar 16000 -sample_fmt s16 {shlex.quote(str(dst))}'
    print("🔁 변환 실행:", cmd)
    subprocess.check_call(cmd, shell=True)
    return str(dst)


# ===== 3) Azure STT (연속 인식) =====
def stt_file(path: str, lang="ko-KR"):
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

    recognizer.start_continuous_recognition()
    while not done:
        time.sleep(0.2)
    recognizer.stop_continuous_recognition()
    return " ".join(texts)

# ===== 실행 예시 =====
if __name__ == "__main__":
    # 여기에 원본 파일 경로만 지정하세요 (확장자 상관없음: wav/mp3/wma 등)
    src_file = "aiaaudio/보험금청구접수.mp3"  # 혹은 "audio/sample.mp3"

    norm_path = ensure_azure_format(src_file)
    print(f"✅ 사용 파일: {norm_path}")
    full_text = stt_file(norm_path, lang="ko-KR")
    print("\n📄 전체 인식 결과:\n", full_text)
