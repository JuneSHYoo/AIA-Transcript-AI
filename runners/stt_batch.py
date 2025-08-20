# batch_run.py
import os, time, json, csv, urllib.parse, datetime, sys
from pathlib import Path
from dotenv import load_dotenv
import requests
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
import re

load_dotenv()
SPEECH_KEY    = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
ST_ACCOUNT    = os.getenv("AZURE_STORAGE_ACCOUNT")
ST_KEY        = os.getenv("AZURE_STORAGE_KEY")
ST_CONTAINER  = os.getenv("AZURE_STORAGE_CONTAINER", "audio")

if not (SPEECH_KEY and SPEECH_REGION and ST_ACCOUNT and ST_KEY and ST_CONTAINER):
    raise SystemExit("환경변수 누락: .env의 AZURE_* 값들을 확인하세요.")

SPEECH_API = f"https://{SPEECH_REGION}.api.cognitive.microsoft.com/speechtotext/v3.2/transcriptions"

def upload_and_get_sas(local_path: str) -> str:
    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(p)

    conn = f"DefaultEndpointsProtocol=https;AccountName={ST_ACCOUNT};AccountKey={ST_KEY};EndpointSuffix=core.windows.net"
    bsc = BlobServiceClient.from_connection_string(conn)
    container = bsc.get_container_client(ST_CONTAINER)
    try:
        container.create_container()
    except Exception:
        pass

    blob = container.get_blob_client(p.name)
    with open(p, "rb") as f:
        blob.upload_blob(f, overwrite=True)

    expiry = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    sas = generate_blob_sas(
        account_name=ST_ACCOUNT,
        container_name=ST_CONTAINER,
        blob_name=p.name,
        account_key=ST_KEY,
        permission=BlobSasPermissions(read=True),
        expiry=expiry,
    )
    return f"https://{ST_ACCOUNT}.blob.core.windows.net/{ST_CONTAINER}/{urllib.parse.quote(p.name)}?{sas}"

def create_batch(content_urls, locale="ko-KR", display="koKR-diarization-demo"):
    headers = {
        "Ocp-Apim-Subscription-Key": SPEECH_KEY,
        "Content-Type": "application/json",
    }

    # 1) v3.2+ 신규 스키마: diarization.speakers.{minCount,maxCount}
    body_new = {
        "displayName": display,
        "locale": locale,
        "contentUrls": content_urls,
        "properties": {
            "timeToLiveHours": 48,                  # 결과 유지 기간(최대 48h)
            "wordLevelTimestampsEnabled": True,     # 단어 타임스탬프
            "punctuationMode": "DictatedAndAutomatic",  # 문장부호 처리
            "profanityFilterMode": "None",          # 비속어 처리: None/Masked/Removed/Tags
            "diarizationEnabled": False,             # 화자 분리 ON
            # "diarization": {
            #     "speakers": { "minCount": 2, "maxCount": 2 }  # 신규 스키마
            # }
            "channels" :[0,1] # L=0, R=1
        }
    }


    r = requests.post(SPEECH_API, headers=headers, data=json.dumps(body_new), timeout=30)
    if r.status_code >= 400:
        print("=== DEBUG create_batch (new schema) error ===")
        print("Status:", r.status_code)
        print("Body  :", r.text)

        # 2) 폴백: 구 스키마 (일부 리전/버전에서 여전히 이 형태만 허용)
        body_legacy = {
            "displayName": display,
            "locale": locale,
            "contentUrls": content_urls,
            "properties": {
                "timeToLiveHours": 48,                  # 결과 유지 기간(최대 48h)
                "wordLevelTimestampsEnabled": True,     # 단어 타임스탬프
                "punctuationMode": "DictatedAndAutomatic",  # 문장부호 처리
                "profanityFilterMode": "None",          # 비속어 처리: None/Masked/Removed/Tags
                "diarizationEnabled": False,             # 화자 분리 ON
                # "diarization": {
                #     "speakers": { "minCount": 2, "maxCount": 2 }  # 신규 스키마
                # }
                "channels" :[0,1] # L=0, R=1
            }
        }
        r2 = requests.post(SPEECH_API, headers=headers, data=json.dumps(body_legacy), timeout=30)
        if r2.status_code >= 400:
            print("=== DEBUG create_batch (legacy schema) error ===")
            print("Status:", r2.status_code)
            print("Body  :", r2.text)
            r2.raise_for_status()
        r = r2  # 성공하면 이걸 사용

    r.raise_for_status()
    job_url = r.headers.get("Location")
    if not job_url:
        raise RuntimeError("작업 URL(Location) 미수신")
    return job_url



def wait_until_done(job_url, poll=5, timeout_min=120):
    hdr = {"Ocp-Apim-Subscription-Key": SPEECH_KEY}
    end = time.time() + 60 * timeout_min
    while True:
        jr = requests.get(job_url, headers=hdr, timeout=30)
        jr.raise_for_status()
        data = jr.json()
        print("Status:", data.get("status"))
        if data.get("status") in ("Succeeded", "Failed"):
            return data
        if time.time() > end:
            raise TimeoutError("배치 작업 시간 초과")
        time.sleep(poll)

def list_result_files(job_data):
    hdr = {"Ocp-Apim-Subscription-Key": SPEECH_KEY}
    files_url = job_data["links"]["files"]
    fr = requests.get(files_url, headers=hdr, timeout=30)
    fr.raise_for_status()
    return fr.json()["values"]

def download_text(url: str, out_path: Path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_text(r.text, encoding="utf-8")

def _to_seconds(val):
    """
    Batch JSON의 시간 표현을 전부 초(float)로 변환:
    - 숫자/숫자문자열(100ns 단위)  ex) 4000000  -> 0.4s
    - ISO8601 ex) 'PT1M40.28S'
    - 시:분:초  ex) '00:01:40.28'
    """
    if val is None:
        return 0.0

    # 숫자형이면: 너무 큰 값(100ns)일 수 있어 보정
    if isinstance(val, (int, float)):
        return float(val) if val < 1e6 else float(val) / 10_000_000.0

    s = str(val).strip()

    # 순수 숫자문자열(100ns)
    if s.isdigit():
        v = int(s)
        return float(v) if v < 1e6 else v / 10_000_000.0

    # ISO8601 'PT..' 형식
    if s.startswith("PT"):
        h = m = 0.0
        sec = 0.0
        mh = re.search(r'(\d+(?:\.\d+)?)H', s)
        mm = re.search(r'(\d+(?:\.\d+)?)M', s)
        ms = re.search(r'(\d+(?:\.\d+)?)S', s)
        if mh: h = float(mh.group(1))
        if mm: m = float(mm.group(1))
        if ms: sec = float(ms.group(1))
        return h*3600 + m*60 + sec

    # 'HH:MM:SS(.fff)'
    m = re.match(r'(\d+):([0-5]?\d):([0-5]?\d(?:\.\d+)?)', s)
    if m:
        hh = int(m.group(1)); mm = int(m.group(2)); ss = float(m.group(3))
        return hh*3600 + mm*60 + ss

    return 0.0


def parse_transcription_json(json_path: Path):
    j = json.loads(json_path.read_text(encoding="utf-8"))
    phrases = j.get("recognizedPhrases") or j.get("combinedRecognizedPhrases") or []

    rows = []
    for ph in phrases:
        # speaker가 없을 수도 있음
        speaker = ph.get("speaker")

        # 우선순위: offsetMilliseconds/durationMilliseconds → offset/duration
        # (밀리초가 있으면 그대로 쓰는 게 가장 간단하고 정확)
        if "offsetMilliseconds" in ph and "durationMilliseconds" in ph:
            start = round(float(ph["offsetMilliseconds"]) / 1000.0, 2)
            duration = round(float(ph["durationMilliseconds"]) / 1000.0, 2)
        else:
            start = round(_to_seconds(ph.get("offset", 0)), 2)
            duration = round(_to_seconds(ph.get("duration", 0)), 2)

        # 텍스트 추출
        text = ""
        nbest = ph.get("nBest") or []
        if nbest:
            text = nbest[0].get("display") or nbest[0].get("lexical") or ""
        else:
            text = ph.get("display") or ph.get("lexical") or ph.get("text") or ""

        rows.append({"speaker": speaker, "start": start, "duration": duration, "text": text})

    # 시간순 정렬
    rows.sort(key=lambda r: (r["start"], r["speaker"] if r["speaker"] is not None else 999))
    return rows

def save_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "speaker", "start_sec", "duration_sec", "text"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["speaker"], r["start"], r["duration"], r["text"]])

if __name__ == "__main__":
    # 사용법:
    #   python batch_run.py            -> convert/*.wav 자동 업로드
    #   python batch_run.py 파일1.wav 파일2.wav ...
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        targets = sorted(Path("../convert").glob("*.wav"))

    if not targets:
        raise SystemExit("업로드할 WAV 파일이 없습니다. convert/*.wav 또는 인자를 지정하세요.")

    print("▶ 업로드")
    urls = [upload_and_get_sas(str(p)) for p in targets]
    for u in urls: print(" -", u)

    print("\n▶ 배치 작업 생성")
    job_url = create_batch(urls, locale="ko-KR", display="koKR-diarization-demo")
    print("Job URL:", job_url)

    print("\n▶ 상태 폴링")
    job_data = wait_until_done(job_url, poll=5, timeout_min=120)
    if job_data["status"] == "Failed":
        print(json.dumps(job_data, ensure_ascii=False, indent=2))
        raise SystemExit("❌ 배치 실패")

    print("\n▶ 결과 다운로드/파싱")
    out_dir = Path("../batch_results"); out_dir.mkdir(exist_ok=True)
    entries = list_result_files(job_data)
    trans = [e for e in entries if e["kind"] == "Transcription"]
    if not trans:
        raise SystemExit("Transcription JSON 링크가 없습니다.")
    json_url = trans[0]["links"]["contentUrl"]
    json_path = out_dir / "transcription.json"
    download_text(json_url, json_path)

    rows = parse_transcription_json(json_path)
    save_csv(rows, out_dir / "transcription_speaker_timeline.csv")

    print("\n✅ 완료")
    print("JSON :", json_path)
    print("CSV  :", out_dir / "transcription_speaker_timeline.csv")
