import os, json, requests,argparse
from dotenv import load_dotenv
from pathlib import Path
import sys

# ===== CLI 인자 파싱 =====
parser = argparse.ArgumentParser(description="Fireworks API inference")
parser.add_argument("--prompt", required=True, help="시스템 프롬프트 파일 경로")
parser.add_argument("--raw-text", required=True, help="콜 전사 텍스트 파일 경로")
args = parser.parse_args()

load_dotenv()
API_KEY = os.getenv("FIREWORKS_API_KEY")
MODEL_ID = os.getenv("FIREWORKS_MODEL_DS_ID")  # 예: gpt-oss-20b

if not API_KEY:
    raise RuntimeError("환경변수 FIREWORKS_API_KEY 가 비어 있습니다.")
if not MODEL_ID:
    raise RuntimeError("환경변수 FIREWORKS_MODEL_GPT_ID 가 비어 있습니다. (예: gpt-oss-20b)")


# ===== 파일 읽기 =====
def load_file(path: str) -> str:
    with open(Path(path), "r", encoding="utf-8") as f:
        return f.read().strip()

SYSTEM_PROMPT = load_file(args.prompt)
RAW_TEXT = load_file(args.raw_text)


messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"아래 콜 전사 텍스트를 분석해 위 지시에 따라 요약해줘.\n<콜 전사>\n{RAW_TEXT}\n</콜 전사>"},
]


url = "https://api.fireworks.ai/inference/v1/chat/completions"
payload = {
    "model": MODEL_ID,
    "messages": messages,
    "max_tokens": 5000,
    "top_p": 0.9,
    "top_k": 40,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "temperature": 0.2,
    "store": False
      # "stream": True
    # "response_format": {"type": "json_object"}   # ← gpt-oss-20b에서는 제거!
}
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}


# resp = requests.post(url, headers=headers, data=json.dumps(payload))
# resp.raise_for_status()
# data = resp.json()

response = requests.post(url, headers=headers, data=json.dumps(payload))
response.raise_for_status()


data = response.json()

choices = data.get("choices") or []
if not choices:
    raise SystemExit("[ERROR] choices가 비어 있습니다.\n" + json.dumps(data, ensure_ascii=False, indent=2))

choice = choices[0]
msg = choice.get("message") or {}

# ⚠️ content가 없을 수 있으므로 안전한 fallback 순서로 추출
content = (
    msg.get("content")                # 보통 여기
    or msg.get("reasoning_content")   # 일부 모델은 여기에만 담음
    or choice.get("text")             # 레거시/일부 엔드포인트
    or data.get("output_text")        # 또 다른 변형
    or ""                             # 최종 실패 시 빈 문자열
)

if not content:
    # 디버깅 도움: 실제 받은 키들을 보여줌
    raise SystemExit(
        "[PARSE ERROR] 본문을 찾지 못했습니다.\n"
        + "keys(message)=" + str(list(msg.keys())) + "\n"
        + "keys(choice)=" + str(list(choice.keys())) + "\n"
        + json.dumps(data, ensure_ascii=False, indent=2)
    )

print("✅ Deepseek 요약 결과\n")
print(content)


