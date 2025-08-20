# common_audio.py
import os, subprocess, shlex, tempfile, json
from pathlib import Path
from typing import Optional, Tuple, Literal, Dict

import numpy as np

# soundfile가 없으면 wave로 폴백
try:
    import soundfile as sf
    _HAS_SF = True
except Exception:
    import wave, struct
    _HAS_SF = False

def run(cmd: str):
    print("RUN:", cmd)
    subprocess.check_call(cmd, shell=True)

def probe_audio(path: str) -> dict:
    cmd = f'ffprobe -v error -show_streams -select_streams a -of json {shlex.quote(path)}'
    out = subprocess.check_output(cmd, shell=True, text=True)
    info = json.loads(out)
    if not info.get("streams"):
        raise RuntimeError(f"오디오 스트림을 찾지 못함: {path}")
    s = info["streams"][0]
    return {
        "codec_name": s.get("codec_name"),
        "sample_rate": int(s.get("sample_rate", "0")),
        "channels": int(s.get("channels", 0)),
        "bits_per_sample": int(s.get("bits_per_sample", 0) or 0),
    }

def ensure_wav_16k_mono(src_path: str, convert_dir: str="convert") -> str:
    Path(convert_dir).mkdir(exist_ok=True)
    src = Path(src_path)
    dst = Path(convert_dir) / f"{src.stem}_16k.wav"

    # 이미 만들어져 있으면 재활용
    if dst.exists():
        return str(dst)

    spec = probe_audio(str(src))
    need = (
        spec["codec_name"] != "pcm_s16le" or
        spec["sample_rate"] != 16000 or
        spec["channels"] != 1 or
        spec["bits_per_sample"] != 16
    )
    if not need and src.suffix.lower() == ".wav":
        # 권장 규격이면 복사
        import shutil
        shutil.copy(str(src), str(dst))
        return str(dst)

    cmd = f'ffmpeg -y -i {shlex.quote(str(src))} -ac 1 -ar 16000 -sample_fmt s16 {shlex.quote(str(dst))}'
    run(cmd)
    return str(dst)

def wma_to_wav(src_path: str, convert_dir: str="convert") -> str:
    Path(convert_dir).mkdir(exist_ok=True)
    src = Path(src_path)
    dst = Path(convert_dir) / f"{src.stem}_wma2wav.wav"
    if dst.exists():
        return str(dst)
    cmd = f'ffmpeg -y -i {shlex.quote(str(src))} -c:a pcm_s16le {shlex.quote(str(dst))}'
    run(cmd)
    return str(dst)

def extract_lr_mono_wav(src_path: str, convert_dir: str="split") -> Tuple[Optional[str], Optional[str]]:
    """스테레오 입력에서 L/R을 각각 16k/mono/pcm_s16le wav로 추출.
       -map_channel 미지원 환경을 고려해 channelsplit 사용
    """
    Path(convert_dir).mkdir(exist_ok=True)
    stem = Path(src_path).stem
    L = Path(convert_dir) / f"{stem}_L_16k.wav"
    R = Path(convert_dir) / f"{stem}_R_16k.wav"

    if L.exists() and R.exists():
        return str(L), str(R)

    # channelsplit로 FL/FR 분리 후 각각 모노/16k/16bit로 출력
    cmd = (
        f"ffmpeg -y -i {shlex.quote(src_path)} "
        f"-filter_complex \"channelsplit=channel_layout=stereo[left][right]\" "
        f"-map \"[left]\"  -ar 16000 -ac 1 -sample_fmt s16 {shlex.quote(str(L))} "
        f"-map \"[right]\" -ar 16000 -ac 1 -sample_fmt s16 {shlex.quote(str(R))}"
    )
    run(cmd)
    return str(L), str(R)


def _read_wav_mono_float(path: str) -> np.ndarray:
    """PCM WAV를 [-1,1] float으로 로드."""
    if _HAS_SF:
        data, _sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data[:,0]
        return data.astype(np.float32, copy=False)

    # wave 모듈 폴백(16-bit 가정)
    import wave, struct
    with wave.open(path, "rb") as w:
        nchan = w.getnchannels()
        sampw = w.getsampwidth()
        nfrm  = w.getnframes()
        frames = w.readframes(nfrm)
    if sampw != 2:
        raise RuntimeError("wave 폴백은 16-bit PCM만 지원")
    vals = np.frombuffer(frames, dtype="<i2")  # little-endian 16-bit
    if nchan == 2:
        vals = vals.reshape(-1,2)[:,0]
    return (vals.astype(np.float32) / 32768.0)

def is_dual_mono_by_samples(stereo_src: str, tmp_dir: str="__tmp_dualmono") -> bool:
    """스테레오 입력에서 L/R 추출 후 파형 차이가 거의 없으면 dual mono."""
    L, R = extract_lr_mono_wav(stereo_src, convert_dir=tmp_dir)
    a = _read_wav_mono_float(L)
    b = _read_wav_mono_float(R)
    n = min(len(a), len(b))
    if n == 0:
        return False
    mad = float(np.max(np.abs(a[:n] - b[:n])))
    print(f"[dual-mono check] max|L-R| = {mad:.8f}")
    return mad < 1e-5

Kind = Literal["mono", "dual_mono", "true_stereo"]

def inspect_and_prepare(src_path: str) -> Dict[str, Optional[str]]:
    """
    입력 파일을 점검/준비하고 분기 판단을 리턴.
    - .wma -> .wav 변환
    - .wav/.mp3 유지
    - 채널 수 확인 후:
        mono        : 권장 규격 16k/mono/wav 생성
        dual_mono   : L/R 동일 → 권장 규격 16k/mono/wav 생성
        true_stereo : 실제 L/R 상이 → split/ 아래 L/R 추출
    returns:
      {
        "kind": "mono"|"dual_mono"|"true_stereo",
        "mono_wav": <path or None>,
        "L_wav": <path or None>,
        "R_wav": <path or None>,
        "orig_path": <str>,
      }
    """
    p = Path(src_path)
    # 1) 확장자 케어
    if p.suffix.lower() == ".wma":
        base = wma_to_wav(str(p))  # wav로 변환
    else:
        base = str(p)

    spec = probe_audio(base)
    if spec["channels"] <= 1:
        mono_wav = ensure_wav_16k_mono(base)
        return {"kind":"mono", "mono_wav": mono_wav, "L_wav": None, "R_wav": None, "orig_path": str(p)}

    # 스테레오
    if is_dual_mono_by_samples(base):
        mono_wav = ensure_wav_16k_mono(base)
        return {"kind":"dual_mono", "mono_wav": mono_wav, "L_wav": None, "R_wav": None, "orig_path": str(p)}
    else:
        L, R = extract_lr_mono_wav(base, convert_dir="../split")
        return {"kind":"true_stereo", "mono_wav": None, "L_wav": L, "R_wav": R, "orig_path": str(p)}

## 테스트 실행문
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"사용법: python {sys.argv[0]} <오디오 파일 경로>")
        sys.exit(1)

    src_file = sys.argv[1]
    result = inspect_and_prepare(src_file)
    print("처리 결과:", result)

