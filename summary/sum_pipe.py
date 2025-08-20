import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description="Fireworks summarizer launcher")
parser.add_argument("--model", required=True, choices=["llama", "gpt"], help="모델 타입 (llama | gpt)")
parser.add_argument("--prompt", required=True, help="시스템 프롬프트 파일 경로")
parser.add_argument("--raw-text", required=True, help="콜 전사 텍스트 파일 경로")
args = parser.parse_args()

# 실행할 파이썬 파일 매핑
MODEL_SCRIPTS = {
    "llama": "firework_llama.py",
    "gpt": "firework_gpt.py",
}

script = MODEL_SCRIPTS[args.model]

# 선택된 스크립트 실행 (CLI 인자 전달)
cmd = [
    sys.executable, script,
    "--prompt", args.prompt,
    "--raw-text", args.raw_text
]

subprocess.run(cmd, check=True)
