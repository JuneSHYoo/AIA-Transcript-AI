import os, json, requests, argparse
from dotenv import load_dotenv
from pathlib import Path

# ===== CLI 인자 파싱 =====
parser = argparse.ArgumentParser(description="Fireworks API inference")
parser.add_argument("--prompt", required=True, help="시스템 프롬프트 파일 경로")
parser.add_argument("--raw-text", required=True, help="콜 전사 텍스트 파일 경로")
args = parser.parse_args()


# ===== 환경 변수 로드 =====
load_dotenv()
API_KEY = os.getenv("FIREWORKS_API_KEY")
MODEL_ID = os.getenv("FIREWORKS_MODEL_ID") # 예: accounts/fireworks/models/llama-v3p3-70b-instruct

# ===== 파일 읽기 =====
def load_file(path: str) -> str:
    with open(Path(path), "r", encoding="utf-8") as f:
        return f.read().strip()

SYSTEM_PROMPT = load_file(args.prompt)
raw_text = load_file(args.raw_text)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"아래 콜 전사 텍스트를 분석해 위 지시에 따라 요약해줘.\n<콜 전사>\n{raw_text}\n</콜 전사>"},
]
resp = requests.post(
    "https://api.fireworks.ai/inference/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    data=json.dumps({
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": 5000,
        "temperature": 0.2,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stream": False,
        "store": False
    }),
    timeout=120,
)
resp.raise_for_status()

print("✅ Meta-Llama-3-70B-Instruct \n")
print(resp.json()["choices"][0]["message"]["content"])
