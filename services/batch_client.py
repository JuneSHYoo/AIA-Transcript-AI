import time
import json
from pathlib import Path
import requests
import datetime as dt

########################################
## 배치 작업 생성/폴링/결과 파일 목록/다운로드  ##
########################################
def create_batch_stereo(
    speech_key: str,
    region: str,
    content_urls,
    locale: str = "ko-KR",
    *,
    channels: list[int] | None = None,   # None=모노, [0,1]=단일 스테레오 파일 분리
    diarization: bool = False,
    display_name: str | None = None,
    diar_min: int | None = None,
    diar_max: int | None = None,
):
    """
    - channels is None  → 모노/단일채널/혹은 L·R을 '각각 파일'로 올리는 경우 (channels 필드 넣지 않음)
    - channels = [0,1] → '하나의 스테레오 파일'을 올리고 Azure에게 L/R 분리를 맡길 때만 사용
    """
    import datetime as _dt

    if display_name is None:
        display_name = f"batch-{'stereo' if channels else 'mono'}-{_dt.datetime.utcnow().isoformat()}"

    # ✅ v3.2 권장 (세부 diar 옵션 지원)
    endpoint = f"https://{region}.api.cognitive.microsoft.com/speechtotext/v3.2/transcriptions"
    headers = {
        "Ocp-Apim-Subscription-Key": speech_key,
        "Content-Type": "application/json",
    }

    # 🔒 방어: channels 사용 조건 검증
    if channels is not None:
        if channels != [0, 1]:
            raise ValueError(f"`channels`는 [0, 1]만 허용합니다. 입력: {channels}")
        if not isinstance(content_urls, (list, tuple)) or len(content_urls) != 1:
            raise ValueError("channels를 사용할 때는 content_urls에 '단 하나의 스테레오 파일'만 넣어야 합니다.")

    # 공통 properties
    properties: dict = {
        "timeToLiveHours": 48,
        "wordLevelTimestampsEnabled": True,
        "displayFormWordLevelTimestampsEnabled": True,
        "punctuationMode": "DictatedAndAutomatic",
        "profanityFilterMode": "Masked",
    }

    # diarization on이면 세부 설정(옵션)
    if diarization:
        properties["diarizationEnabled"] = True
        if diar_min is not None or diar_max is not None:
            properties["diarization"] = {
                "speakers": {
                    "minCount": diar_min if diar_min is not None else 2,
                    "maxCount": diar_max if diar_max is not None else 2,
                }
            }

    # ⭐ 단일 스테레오 파일일 때만 channels 추가
    if channels is not None:
        properties["channels"] = channels  # [0, 1]

    body = {
        "displayName": display_name,
        "locale": locale,
        "contentUrls": content_urls,
        "properties": properties,
    }

    r = requests.post(endpoint, headers=headers, json=body, timeout=60)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # 에러 디버깅 편의: 응답 본문도 같이 보여주기
        raise RuntimeError(f"Azure STT batch create failed: {e}\nRESPONSE: {r.text}") from e

    job_url = r.headers.get("Location")
    if not job_url:
        # Location 헤더 누락 시에도 바로 원인 추적 가능하게
        raise RuntimeError(f"작업 URL(Location) 미수신\nRESPONSE: {r.status_code} {r.text}")
    return job_url


def create_batch_mono_diar(speech_key: str, region: str, content_urls, locale="ko-KR", display="koKR-mono-diar", min_spk=2, max_spk=2):
    """모노 파일 전제: diarization 켬"""
    api = f"https://{region}.api.cognitive.microsoft.com/speechtotext/v3.2/transcriptions"
    headers = {"Ocp-Apim-Subscription-Key": speech_key, "Content-Type": "application/json"}
    body = {
        "displayName": display,
        "locale": locale,
        "contentUrls": content_urls,
        "properties": {
            "timeToLiveHours": 48,
            "wordLevelTimestampsEnabled": True,
            "displayFormWordLevelTimestampsEnabled": True,  # (선택)
            "punctuationMode": "DictatedAndAutomatic",
            "profanityFilterMode": "None",
            "diarizationEnabled": True,
            "diarization": {
                "speakers": {"minCount": min_spk, "maxCount": max_spk}
            }
        }
    }
    r = requests.post(api, headers=headers, data=json.dumps(body), timeout=30)
    if r.status_code >= 400:
        print("=== DEBUG create_batch_mono_diar error ===")
        print("Status:", r.status_code)
        print("Body  :", r.text)
    r.raise_for_status()
    job_url = r.headers.get("Location")
    if not job_url:
        raise RuntimeError("작업 URL(Location) 미수신")
    return job_url

def wait_until_done(job_url: str, speech_key: str, poll=5, timeout_min=120):
    hdr = {"Ocp-Apim-Subscription-Key": speech_key}
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

def list_result_files(job_data: dict, speech_key: str):
    hdr = {"Ocp-Apim-Subscription-Key": speech_key}
    files_url = job_data["links"]["files"]
    fr = requests.get(files_url, headers=hdr, timeout=30)
    fr.raise_for_status()
    return fr.json()["values"]

def download_text(url: str, out_path: Path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_text(r.text, encoding="utf-8")
