import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# .env 파일 로드
load_dotenv()

speech_key = os.getenv("AZURE_SPEECH_KEY")
service_region = os.getenv("AZURE_SPEECH_REGION")

if not speech_key or not service_region:
    raise ValueError("Azure Speech Key 또는 Region이 .env에 설정되지 않았습니다.")

# Speech 설정
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_recognition_language = "ko-KR"

# 마이크 입력 인식
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
print("🎤 말씀하세요...")

result = speech_recognizer.recognize_once()

if result.reason == speechsdk.ResultReason.RecognizedSpeech:
    print("인식 결과:", result.text)
elif result.reason == speechsdk.ResultReason.NoMatch:
    print("❌ 음성을 인식할 수 없습니다.")
else:
    print("⚠ 오류:", result.reason)