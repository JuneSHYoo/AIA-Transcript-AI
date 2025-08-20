# file: hf_check.py
import os
from huggingface_hub import HfApi, InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"

import huggingface_hub, sys
print("huggingface_hub version:", huggingface_hub.__version__)
print("python:", sys.executable)
print("HF_TOKEN set?:", bool(HF_TOKEN))

api = HfApi(token=HF_TOKEN)

info = api.model_info(MODEL_ID, expand=["cardData"])
print("\n=== Basic Info ===")
print("id:", getattr(info, "id", None))
print("private:", getattr(info, "private", None))
print("gated:", getattr(info, "gated", None))
print("sha:", getattr(info, "sha", None))

print("\n=== Card Data (subset) ===")
card = getattr(info, "cardData", {}) or {}
for k in ["license", "tags", "datasets"]:
    print(k, ":", card.get(k))

print("\n=== Inference API test (hf-inference) ===")
try:
    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN, provider="hf-inference")
    resp = client.text_generation("Hello", max_new_tokens=5)
    print("OK:", resp)
except Exception as e:
    print("ERROR:", repr(e))

# 참고: 70B가 서버리스 미제공이면 404가 정상입니다. 이 경우 8B로 테스트:
ALT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
print("\n=== Fallback test with 8B (hf-inference) ===")
try:
    client2 = InferenceClient(model=ALT_MODEL, token=HF_TOKEN, provider="hf-inference")
    resp2 = client2.text_generation("Hello", max_new_tokens=5)
    print("OK:", resp2)
except Exception as e:
    print("ERROR:", repr(e))