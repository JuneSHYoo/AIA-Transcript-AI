import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description="Fireworks summarizer launcher")
parser.add_argument("--model", choices=["llama", "gpt", "ds", "phi"], help="모델 타입 (llama | gpt | ds | phi)")
parser.add_argument("--prompt", required=True, help="시스템 프롬프트 파일 경로")
parser.add_argument("--raw-text", required=True, help="콜 전사 텍스트 파일 경로")
args = parser.parse_args()

# 실행할 파이썬 파일 매핑
MODEL_SCRIPTS = {
    "llama": "firework_llama.py",
    "gpt": "firework_gpt.py",
    "ds" : "firework_ds.py"
    # ,
    # "phi" : "azure_phi.py"
}

# 실행 대상 모델 리스트 결정
if args.model:
    targets = [args.model]
else:
    targets = list(MODEL_SCRIPTS.keys())

# 각 스크립트 실행
for model in targets:
    script = MODEL_SCRIPTS[model]
    print(f"\n🚀 Running {model} summarizer...\n")
    cmd = [
        sys.executable, script,
        "--prompt", args.prompt,
        "--raw-text", args.raw_text
    ]
    subprocess.run(cmd, check=True)
