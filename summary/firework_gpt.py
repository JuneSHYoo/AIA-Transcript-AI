import os, json, requests,argparse
from dotenv import load_dotenv
from pathlib import Path

# ===== CLI 인자 파싱 =====
parser = argparse.ArgumentParser(description="Fireworks API inference")
parser.add_argument("--prompt", required=True, help="시스템 프롬프트 파일 경로")
parser.add_argument("--raw-text", required=True, help="콜 전사 텍스트 파일 경로")
args = parser.parse_args()

load_dotenv()
API_KEY = os.getenv("FIREWORKS_API_KEY")
MODEL_ID = os.getenv("FIREWORKS_MODEL_GPT_ID")  # 예: gpt-oss-20b

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

# SYSTEM_PROMPT = (
#     "[시스템 지시]\n"
#     "당신은 고객센터 콜 전사 데이터를 분석하는 전문 AI입니다.\n"
#     "입력되는 텍스트는 고객과 상담사의 대화이지만, 화자 분리가 되어 있지 않고 발화가 연속적으로 기록되어 있습니다.\n"
#     "당신의 임무는 이 텍스트에서 고객 질문, 상담사 답변, 전체 상담 내용을 각각 정확하게 요약하는 것입니다.\n"
#     "요약 정확도는 95% 이상을 목표로 하며, 불필요한 추측이나 추가 설명은 금지합니다.\n"
#     "사고과정/중간추론은 출력하지 말고 최종 결과만 내세요.\n"
#     "반드시 아래 JSON 스키마로만 출력하세요(설명 금지).\n"
#     "출력 스키마:\n"
#     "{"
#     "\"고객 질문 요약\": \"...\","
#     "\"상담사 답변 요약\": \"...\","
#     "\"전체 상담 요약\": \"...\""
#     "}"
# )
#
# GUIDE = (
#     "[작업 규칙]\n"
#     "-'문의, 궁금하다, 알려주세요'등 질문·의문 형태는 고객 발화로 간주합니다.\n"
#     "-'네 고객님, 안내드리겠습니다, ~됩니다, ~필요합니다' 등 설명·안내·응답 형태는 상담사 발화로 간주합니다.\n"
#     "- 문장이 이어져 있을 경우, 질문 → 응답 → 추가 질문 → 응답과 같은 QA 패턴을 추론하여 화자를 구분합니다.\n"
#     "- 화자 구분이 모호한 경우, 문맥상 가장 자연스러운 질문/답변 역할에 맞추어 분리합니다.\n"
#     "- 새로운 화자의 개입 없이 동일 주제 설명이 이어지면 같은 화자로 처리합니다."
#     "[작업 방법]\n"
#     "1. 전체 텍스트를 위 규칙에 따라 고객 발화와 상담사 발화로 구분합니다."
#     "2. 고객의 질문 요지를 압축하여 요약합니다."
#     "3. 상담사의 답변 내용을 압축하여 요약합니다."
#     "4. 전체 상담의 맥락과 결론을 간단히 요약합니다."
#     "5. 반드시 핵심 의미를 유지하고, 불필요한 부연이나 상상은 하지 않습니다."
# )
#
# # ➊ 여기 실제 전사 본문을 넣으세요.
# raw_text = """여보세요 네, 안녕하세요. 그냥 여기 AI 생각입니다. 김영희 고객님, 계약자분 본인 맞으세요? 고객님, 용종 수술하셔서 청구 문의 전화 주셨던 게 맞으시죠? 네 네 혹시 어느 부분 용종 수술하셨던 거예요? 어 그 왜 뭐지? 저기랑 다 됐어요 뭐니? 밀양계장 수술을 언제 하셨었어요, 고객님? 어제 했어요. 내시경 검사하시는 도중에 선사를 하셨던 게 맞으세요? 예. 현재 103 건강 보험 안에 수소 특약에 가입이 되어있고 60 1-1 번씩 수술비 청구가 가능하셔서 제가 청구를 했을 때 어떤 서류 준비를 하셔야 되시는지 고객님 휴대폰 문자 메시지를 보내드릴 거고요. 네 네 서류 준비됐으면 접수를 하셔야 되잖아요. 네네. 전화로 지금 몇 가지를 여쭙고 접수를 해주면. 통화 진행 후에 고객님께 카카오톡 하나 발송이 되는데 그 서류 준비하셔서 카카오톡에 들어가서 사진 찍어서 병원 서류 1장 올려주시면 되시는데 가능하세요? 네, 그러면 그 엄 목요일날 다시 이제 혹시 그 결과보러 가야 되니까 그때 보낼게요 그럼. 그럼 제가 몇 가지 여쭤고 전화로 접수를 해드릴 테니 고객님이 직접과 실제 거주를 하고 계시는 곳은 한국 맞으시죠? 네, 저희 접수가 되거나 지급을 해드릴 때 고객님께 접수됐습니다. 지급됐습니다. 이렇게 안내를 해드려야 되는데 안내를 받으실 때 혹시 문자로 안내해 드려도 괜찮으세요? 습니다. 카톡으로 보내주시면 카톡으로 보내드릴 거고 보험금 지급받으실 수 있는 은행 어느 은행으로 신청해 드릴까요? 농협이요, 지금 보험 빠져나간 농협으로 해주시면 될 것 같습니다. 계좌번호를 말씀해 주시겠어요? 예 예 말씀해 주셨던 계좌 고객님, 본인 계좌로 확인되셔서 제가 전화 접수해 드렸고 통화 종료에 카카오톡 하나 발송이 되거든요. 토요일날 병원에 가셔서 어떤 서류 필요해서 문자 보내드렸으니까 서류 준비하셔서 그 카카오톡에 들어가서 사진 찍어서 올려주시면 되세요? 보험은 나옵니까 수습비가? 예 수출이 청구는 가능하시고 정확한 금액은 심사후에 결정이 되시긴 하는데 현재는 이종 수술로 삼십만원 확인되세요. 감사합니다. 감사합니다. 피해자의 부족한 상품 안내서를 한 번씩 드리고 있어서 시간 되실 때 통화 부탁드리겠습니다. 고맙습니다. 감사합니다. 건강하십시오."""
#
# messages = [
#     {"role": "system", "content": SYSTEM_PROMPT},
#     {"role": "user",
#      "content": f"{GUIDE}\n\n아래 콜 전사 텍스트를 분석해 위 지시에 따라 요약해줘. "
#                 f"JSON만 출력하세요.\n<콜 전사>\n{raw_text}\n</콜 전사>"}
# ]

url = "https://api.fireworks.ai/inference/v1/chat/completions"
payload = {
    "model": MODEL_ID,
    "messages": messages,
    "max_tokens": 1500,   # 넉넉하게
    "temperature": 0.2,
    "top_p": 0.9,
    # "response_format": {"type": "json_object"}   # ← gpt-oss-20b에서는 제거!
}
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
# print("status:", resp.status_code)

data = resp.json()
# print(json.dumps(data, indent=2, ensure_ascii=False))  # 구조 확인용

# 안전 파서: content → reasoning_content → text 순서로 시도
msg = (data.get("choices") or [{}])[0].get("message", {})
content = msg.get("content")
if not content:
    content = msg.get("reasoning_content") or (data["choices"][0].get("text"))

# content 안에서 첫 번째 JSON 오브젝트만 추출(혹시 앞에 잡담 섞이면)
def extract_first_json(s: str) -> str:
    if not s:
        return ""
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if ch == '{':
            if depth == 0: start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start != -1:
                return s[start:i+1]
    return s  # 못 찾으면 원문 반환

json_text = extract_first_json(content or "")

print("✅ gpt-oss-20B 요약 결과 \n")
try:
    parsed = json.loads(json_text)
    print("\n=== 요약(JSON) ===")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
except Exception:
    print("\n=== 원문 ===")
    print(content or "[empty]")
