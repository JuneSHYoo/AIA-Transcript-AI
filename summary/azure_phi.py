import os, json, requests,argparse
from dotenv import load_dotenv
from pathlib import Path
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential


# ===== CLI 인자 파싱 =====
parser = argparse.ArgumentParser(description="Fireworks API inference")
parser.add_argument("--prompt", required=True, help="시스템 프롬프트 파일 경로")
parser.add_argument("--raw-text", required=True, help="콜 전사 텍스트 파일 경로")
args = parser.parse_args()

load_dotenv()
ENDPOINT = os.getenv("AZURE_FOUNDRY_ENDPOINT")    # https://<project>.<region>.inference.ai.azure.com
API_KEY  = os.getenv("AZURE_FOUNDRY_API_KEY")
MODEL    = os.getenv("AZURE_FOUNDRY_MODEL")

if not API_KEY:
    raise RuntimeError("환경변수 AZURE_FOUNDRY_API_KEY 가 비어 있습니다.")
if not MODEL:
    raise RuntimeError("환경변수 AZURE_FOUNDRY_MODEL 가 비어 있습니다. (예: gpt-oss-20b)")


# ===== 파일 읽기 =====
def load_file(path: str) -> str:
    with open(Path(path), "r", encoding="utf-8") as f:
        return f.read().strip()

SYSTEM_PROMPT = load_file(args.prompt)
RAW_TEXT = load_file(args.raw_text)


messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    UserMessage(content=f"아래 콜 전사 텍스트를 분석해 위 지시에 따라 요약해줘.\n<콜 전사>\n{RAW_TEXT}\n</콜 전사>"),
]

endpoint = "https://aiastt-ai.services.ai.azure.com/models"
api_key = os.getenv("AZURE_FOUNDRY_API_KEY")  # 포털에서 복사한 키를 .env에 넣어둠
model_name = "Phi-4"   # 배포 이름이 이대로면 그대로 사용

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(api_key),
    api_version="2024-05-01-preview"
)

response = client.complete(
    messages=messages,
    model=model_name,
    max_tokens=5000,
    temperature=0.2,
    top_p = 0.9,
)

print("✅ phi 요약 결과\n")
print(response.choices[0].message.content)





