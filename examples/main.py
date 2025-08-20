import os
import time
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# .env 로드
load_dotenv()

speech_key = os.getenv("AZURE_SPEECH_KEY")
service_region = os.getenv("AZURE_SPEECH_REGION")

if not speech_key or not service_region:
    raise ValueError("Azure Speech Key 또는 Region이 .env에 설정되지 않았습니다.")

# Speech 설정
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_recognition_language = "ko-KR"

# 오디오 파일 설정 (5분짜리 WAV 파일 경로)
audio_input = speechsdk.AudioConfig(filename="aiaaudio/계좌변경.wav")

# Recognizer 생성
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

# 결과 저장용
recognized_texts = []
done = False

# 인식된 결과 이벤트
def handle_recognized(evt):
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
        print(f"🔹 인식됨: {evt.result.text}")
        recognized_texts.append(evt.result.text)

# 세션 종료 이벤트
def stop_cb(evt):
    global done
    print("🛑 인식 종료")
    done = True

speech_recognizer.recognized.connect(handle_recognized)
speech_recognizer.session_stopped.connect(stop_cb)
speech_recognizer.canceled.connect(stop_cb)

# 연속 인식 시작
speech_recognizer.start_continuous_recognition()
while not done:
    time.sleep(0.5)
speech_recognizer.stop_continuous_recognition()

# 전체 텍스트 출력
full_text = " ".join(recognized_texts)
print("\n📄 전체 인식 결과:")
print(full_text)
