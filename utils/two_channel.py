import soundfile as sf
import numpy as np

data, sr = sf.read("aiaaudio/해약방어.mp3")  # shape: (samples, channels)
if data.ndim == 2 and data.shape[1] == 2:
    diff = np.abs(data[:,0] - data[:,1])
    max_diff = np.max(diff)
    if max_diff < 1e-5:
        print("⚠ L/R 채널 내용이 동일 (dual mono)")
    else:
        print("✅ L/R 채널에 다른 신호가 있음")
else:
    print("모노 파일입니다.")