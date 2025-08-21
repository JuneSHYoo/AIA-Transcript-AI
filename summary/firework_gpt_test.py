import os, json, requests, argparse, re
from dotenv import load_dotenv
from pathlib import Path

FINAL_BEGIN = "<final>"
FINAL_END = "</final>"

def extract_between_markers(text: str, begin: str = FINAL_BEGIN, end: str = FINAL_END) -> str:
    if not text:
        return ""
    m = re.search(rf"{re.escape(begin)}(.*?){re.escape(end)}", text, flags=re.DOTALL)
    return (m.group(1).strip() if m else "")

def load_file(path: str) -> str:
    with open(Path(path), "r", encoding="utf-8") as f:
        return f.read().strip()

parser = argparse.ArgumentParser(description="Fireworks API inference")
parser.add_argument("--prompt", required=True, help="시스템 프롬프트 파일 경로")
parser.add_argument("--raw-text", required=True, help="콜 전사 텍스트 파일 경로")
parser.add_argument("--model", help="모델 ID (환경변수보다 우선)")
args = parser.parse_args()

load_dotenv()
API_KEY = os.getenv("FIREWORKS_API_KEY")
if not API_KEY:
    raise RuntimeError("환경변수 FIREWORKS_API_KEY 가 비어 있습니다.")

MODEL_ID = args.model or os.getenv("FIREWORKS_MODEL_GPT_ID") or "gpt-oss-20b"

base_prompt = load_file(args.prompt)
raw_text = load_file(args.raw_text)

SYSTEM_PROMPT = f"""{base_prompt}

당신은 고객센터 콜 전사 데이터를 분석하는 전문 AI입니다.
사고과정/중간추론/해설을 출력하지 말고, 반드시 최종 요약만 출력하세요.
최종 요약은 반드시 아래 두 태그 사이에만 작성하세요:

{FINAL_BEGIN}
(최종 요약만, 한국어로)
{FINAL_END}

{FINAL_BEGIN} 태그 밖에는 어떤 내용도 쓰지 마세요.
"""
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"아래 콜 전사 텍스트를 분석해 위 지시에 따라 요약해줘.\n<콜 전사>\n{raw_text}\n</콜 전사>"},
]

url = "https://api.fireworks.ai/inference/v1/chat/completions"
payload = {
    "model": MODEL_ID,
    "messages": messages,
    "max_tokens": 1500,
    "temperature": 0.2,
    "top_p": 0.9,
    "stop": [FINAL_END],   # END 태그에서 바로 컷
}
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
resp.raise_for_status()
data = resp.json()

choices = data.get("choices") or []
if not choices:
    raise SystemExit("[ERROR] choices가 비어 있습니다.\n" + json.dumps(data, ensure_ascii=False, indent=2))

msg = choices[0].get("message") or {}
content = msg.get("content") or choices[0].get("text") or data.get("output_text") or ""

# stop으로 잘렸기 때문에 content == 최종 요약 본문
final_body = content.strip()
if not final_body:
    raise SystemExit("[ERROR] 최종 본문이 비어 있습니다. 프롬프트/모델을 확인하세요.")

print(f"✅ {MODEL_ID} 요약 결과\n")
print(f"{FINAL_BEGIN}\n{final_body}\n{FINAL_END}")
