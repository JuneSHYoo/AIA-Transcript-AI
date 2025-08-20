import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from utils.blob_utils import upload_and_get_sas
from services.batch_client import create_batch_stereo, wait_until_done, list_result_files, download_text
from utils.parsers import parse_transcription_json, save_csv

load_dotenv()
SPEECH_KEY    = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
ST_ACCOUNT    = os.getenv("AZURE_STORAGE_ACCOUNT")
ST_KEY        = os.getenv("AZURE_STORAGE_KEY")
ST_CONTAINER  = os.getenv("AZURE_STORAGE_CONTAINER", "audio")

###############################################
## 스테레오(L/R) 멀티채널 전사 (diarization 꺼짐)  ##
###############################################
if not (SPEECH_KEY and SPEECH_REGION and ST_ACCOUNT and ST_KEY and ST_CONTAINER):
    raise SystemExit("환경변수 누락: .env의 AZURE_* 값들을 확인하세요.")

if __name__ == "__main__":
    # 사용법:
    #   python batch_run_stereo.py            -> convert/*.wav 업로드
    #   python batch_run_stereo.py 파일1.wav 파일2.wav ...
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        targets = sorted(Path("../convert").glob("*.wav"))

    if not targets:
        raise SystemExit("업로드할 WAV 파일이 없습니다. convert/*.wav 또는 인자를 지정하세요.")

    print("▶ 업로드")
    urls = [upload_and_get_sas(str(p), ST_ACCOUNT, ST_KEY, ST_CONTAINER) for p in targets]
    for u in urls: print(" -", u)

    print("\n▶ 배치 작업 생성 (스테레오 멀티채널)")
    job_url = create_batch_stereo(SPEECH_KEY, SPEECH_REGION, urls, locale="ko-KR")
    print("Job URL:", job_url)

    print("\n▶ 상태 폴링")
    job_data = wait_until_done(job_url, SPEECH_KEY, poll=5, timeout_min=120)
    if job_data["status"] == "Failed":
        import json
        print(json.dumps(job_data, ensure_ascii=False, indent=2))
        raise SystemExit("❌ 배치 실패")

    print("\n▶ 결과 다운로드/파싱")
    out_dir = Path("../batch_results"); out_dir.mkdir(exist_ok=True)
    entries = list_result_files(job_data, SPEECH_KEY)

    # 에러 파일 링크(있으면 출력)
    for e in entries:
        if e.get("kind") == "Error":
            print("⚠ Error file:", e.get("name"), "->", e["links"]["contentUrl"])

    trans_entries = [e for e in entries if e["kind"] == "Transcription"]
    if not trans_entries:
        raise SystemExit("Transcription JSON 링크가 없습니다.")

    all_rows = []
    # 채널→화자 라벨(원하면 커스터마이즈)
    channel_labels = {0: "Agent", 1: "Customer"}

    for i, ent in enumerate(trans_entries, 1):
        json_url = ent["links"]["contentUrl"]
        json_path = out_dir / f"transcription_{i}.json"
        download_text(json_url, json_path)
        # rows = parse_transcription_json(json_path, stereo_mode=True, channel_labels=channel_labels)
        stereo = bool(channels)  # channels=None → False, [0,1] → True
        if stereo:
            rows = parse_transcription_json(json_path, stereo_mode=True, channel_labels=channel_labels)
        else:
            rows = parse_transcription_json(json_path, stereo_mode=False)  # 모노는 라벨 불필요

        all_rows.extend(rows)

    # save_csv(all_rows, out_dir / "transcription_speaker_timeline.csv")
    for i, ent in enumerate(trans_entries, 1):
        json_url = ent["links"]["contentUrl"]
        json_path = out_dir / f"transcription_{i}.json"
        download_text(json_url, json_path)
        rows = parse_transcription_json(json_path, stereo_mode=True, channel_labels=channel_labels)
        all_rows.extend(rows)

        # 파일명 생성
        if i <= len(targets):
            csv_name = f"{targets[i - 1].stem}_speaker_timeline.csv"
        else:
            csv_name = f"transcription_{i}_speaker_timeline.csv"

        save_csv(rows, out_dir / csv_name)
        print(f"CSV 저장 완료: {csv_name}")

    print("\n✅ 완료")
