# batch_run.py
import os
import json  # ← 추가
import argparse
import time                               # 추가: 시간측정
from datetime import datetime, timezone   # ISO 타임스탬프
from pathlib import Path
from dotenv import load_dotenv
import sys

# 병렬 폴링/다운로드
from concurrent.futures import ThreadPoolExecutor, as_completed  # [ADDED] 병렬 폴링/다운로드

from utils.blob_utils import upload_and_get_sas
from services.batch_client import create_batch_stereo, wait_until_done, list_result_files, download_text
from utils.parsers import parse_transcription_json, save_csv
# 공통 모듈
from common.common_audio import inspect_and_prepare
from common.common_timing import Timer
import shlex, subprocess  # 병합 스크립트 호출용

load_dotenv()
SPEECH_KEY    = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
ST_ACCOUNT    = os.getenv("AZURE_STORAGE_ACCOUNT")
ST_KEY        = os.getenv("AZURE_STORAGE_KEY")
ST_CONTAINER  = os.getenv("AZURE_STORAGE_CONTAINER", "audio")

###############################################
## 모노/스테레오 겸용 배치 전사 (분기 자동) ##
###############################################
if not (SPEECH_KEY and SPEECH_REGION and ST_ACCOUNT and ST_KEY and ST_CONTAINER):
    raise SystemExit("환경변수 누락: .env의 AZURE_* 값들을 확인하세요.")

def parse_args():
    p = argparse.ArgumentParser(description="Azure Speech Batch (auto/mono/stereo)")
    p.add_argument("files", nargs="*", help="업로드할 파일 목록 (미지정 시 convert/*.wav 검색)")
    p.add_argument("--mode", choices=["auto", "mono", "stereo"], default="auto", help="처리 모드 (기본 auto)")
    p.add_argument("--allow-mp3", action="store_true", help="convert/*.mp3 도 업로드 대상에 포함")
    p.add_argument("--locale", default="ko-KR", help="인식 언어 (기본 ko-KR)")
    p.add_argument("--dialog", action="store_true", help="스테레오 시 병합 대화록까지 생성")
    p.add_argument("--txt", action="store_true", help="병합 대화록을 TXT로만 저장")
    p.add_argument("--txt-no-timestamps", action="store_true", help="TXT에서 타임스탬프 제거")
    p.add_argument("--mono-txt", action="store_true", help="모노(diarization) 결과를 TXT로 저장")
    p.add_argument("--mono-txt-no-timestamps", action="store_true", help="모노 TXT에서 타임스탬프 제거")
    return p.parse_args()

def _safe_base(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in name)

def save_txt(rows, path, with_ts=True):
    """rows(list[dict]) → 한 줄씩 TXT 저장
       - with_ts=True: [HH:MM:SS.mmm–HH:MM:SS.mmm] 프리픽스 포함
       - speaker/channel 키가 있으면 'SPEAKER: ' 프리픽스 포함
    """
    def _fmt_ts(sec):
        if sec is None:
            return ""
        t = float(sec)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t - h*3600 - m*60
        return f"{h:02}:{m:02}:{s:06.3f}"

    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            text = (r.get("text") or "").strip()
            spk  = r.get("speaker") or r.get("channel") or ""
            if with_ts and ("start" in r):
                start = _fmt_ts(r.get("start"))
                end   = _fmt_ts(r.get("end"))
                ts = f"[{start}–{end}] " if end else f"[{start}] "
            else:
                ts = ""
            prefix = f"{spk}: " if spk else ""
            f.write(f"{ts}{prefix}{text}\n")

if __name__ == "__main__":
    args = parse_args()

    # 대상 파일 수집
    if args.files:
        targets = [Path(p) for p in args.files]
    else:
        targets = sorted(Path("convert").glob("*.wav"))
        if args.allow_mp3:
            targets += sorted(Path("convert").glob("*.mp3"))

    if not targets:
        raise SystemExit("업로드할 오디오 파일이 없습니다. convert/*.wav (또는 인자를 지정)")

    out_dir = Path("batch_results")
    out_dir.mkdir(exist_ok=True)

    # [MODIFIED] 파일 단위로 전처리/분기/업로드/배치/저장 수행
    for src in targets:
        print(f"\n=== 배치 처리: {src} ===")

        # 0) 파일별 전처리 & 채널 판별 (WMA→WAV, dual_mono 감지, true_stereo 분기 준비)
        info = inspect_and_prepare(str(src))  # {kind, mono_wav, L_wav, R_wav, orig_path}

        # 모드 결정
        if args.mode == "mono":
            do_stereo = False
        elif args.mode == "stereo":
            do_stereo = (info["kind"] == "true_stereo")
            if not do_stereo:
                print("⚠ true_stereo가 아니지만 --mode stereo가 지정되었습니다. mono로 진행합니다.")
        else:  # auto
            do_stereo = (info["kind"] == "true_stereo")

        base = _safe_base(Path(src).stem)

        # 1) 시간 측정 시작
        with Timer():  # [ADDED] 시작/종료/경과 자동 출력
            if not do_stereo:
                # ----------------------------
                # [MONO / DUAL_MONO] 단일 파일 배치
                # ----------------------------
                upload_path = info["mono_wav"]
                if not upload_path:
                    # (안전장치) 없으면 원본 그대로 업로드 시도
                    upload_path = str(src)

                print("▶ 업로드 (mono)")
                url = upload_and_get_sas(upload_path, ST_ACCOUNT, ST_KEY, ST_CONTAINER)
                print(" -", url)

                print("\n▶ 배치 작업 생성 (모노, diarization=True)")
                client_start_iso = datetime.now(timezone.utc).isoformat()  # (참고) 필요시 활용
                t0 = time.perf_counter()

                # [MODIFIED] 모노: channels=None, diarization=True
                job_url = create_batch_stereo(
                    SPEECH_KEY, SPEECH_REGION, [url],
                    locale=args.locale,
                    channels=None,
                    diarization=True,
                    diar_min = 2, diar_max = 2
                )
                print("Job URL:", job_url)

                print("\n▶ 상태 폴링")
                job_data = wait_until_done(job_url, SPEECH_KEY, poll=5, timeout_min=120)
                if job_data.get("status") == "Failed":
                    import json as _json

                    print(_json.dumps(job_data, ensure_ascii=False, indent=2))
                    raise SystemExit("❌ 배치 실패")

                print("\n▶ 결과 다운로드/파싱")
                entries = list_result_files(job_data, SPEECH_KEY)

                # 에러 파일 링크(있으면 출력)
                for e in entries:
                    if e.get("kind") == "Error":
                        print("⚠ Error file:", e.get("name"), "->", e["links"]["contentUrl"])

                trans_entries = [e for e in entries if e.get("kind") == "Transcription"]
                if not trans_entries:
                    raise SystemExit("Transcription JSON 링크가 없습니다.")

                # 단일 파일이므로 첫번째만 사용
                ent = trans_entries[0]
                json_url = ent["links"]["contentUrl"]

                # 1) 임시 저장 후 고정 이름으로 교체
                tmp_path = out_dir / f"_{base}_tmp_mono.json"
                download_text(json_url, tmp_path)

                mono_json_path = out_dir / f"{base}_batch_mono.json"  # [ADDED] 명세화
                tmp_path.replace(mono_json_path)
                print(f"JSON 저장 완료: {mono_json_path.name}")

                # 2) 파싱/CSV/JSON(rows) 저장
                rows = parse_transcription_json(mono_json_path, stereo_mode=False)
                csv_name = out_dir / f"{base}_batch_mono.csv"
                save_csv(rows, csv_name)
                print(f"CSV 저장 완료: {csv_name.name}")

                rows_json = out_dir / f"{base}_batch_mono_rows.json"
                with open(rows_json, "w", encoding="utf-8") as jf:
                    json.dump(rows, jf, ensure_ascii=False, indent=2)
                print(f"JSON 저장 완료: {rows_json.name}")

                if args.mono_txt:
                    txt_name = out_dir / f"{base}_batch_mono.txt"
                    save_txt(rows, txt_name, with_ts=not args.mono_txt_no_timestamps)
                    print(f"TXT 저장 완료: {txt_name.name}")


            else:
                # ----------------------------
                # [TRUE_STEREO] L/R 각각 배치
                # ----------------------------
                L_wav, R_wav = info["L_wav"], info["R_wav"]
                if not (L_wav and R_wav):
                    print("⚠ true_stereo로 판정됐지만 L/R 분리 파일이 없습니다. mono로 처리합니다.")
                    continue

                print("▶ 업로드 (stereo L/R 개별)")
                # urlL = upload_and_get_sas(L_wav, ST_ACCOUNT, ST_KEY, ST_CONTAINER)
                # urlR = upload_and_get_sas(R_wav, ST_ACCOUNT, ST_KEY, ST_CONTAINER)
                # print(" - L:", urlL)
                # print(" - R:", urlR)
                print("▶ 업로드 (stereo L/R 개별)")
                with ThreadPoolExecutor(max_workers=2) as ex:
                    futL = ex.submit(upload_and_get_sas, L_wav, ST_ACCOUNT, ST_KEY, ST_CONTAINER)
                    futR = ex.submit(upload_and_get_sas, R_wav, ST_ACCOUNT, ST_KEY, ST_CONTAINER)
                    urlL, urlR = futL.result(), futR.result()
                print(" - L:", urlL)
                print(" - R:", urlR)

                print("\n▶ 배치 작업 생성 (L/R 각각, diarization=True)")
                # [ADDED] L Job (diarization=True)
                jobL = create_batch_stereo(
                    SPEECH_KEY, SPEECH_REGION, [urlL],
                    locale=args.locale, channels=None, diarization=False
                )
                # [ADDED] R Job (diarization=True)
                jobR = create_batch_stereo(
                    SPEECH_KEY, SPEECH_REGION, [urlR],
                    locale=args.locale, channels=None, diarization=False
                )
                print("Job L:", jobL)
                print("Job R:", jobR)

                # print("\n▶ 상태 폴링 (L)")
                # jdL = wait_until_done(jobL, SPEECH_KEY, poll=5, timeout_min=120)
                # if jdL.get("status") == "Failed":
                #     import json as _json
                #
                #     print(_json.dumps(jdL, ensure_ascii=False, indent=2))
                #     raise SystemExit("❌ 배치 실패(L)")
                #
                # print("\n▶ 상태 폴링 (R)")
                # jdR = wait_until_done(jobR, SPEECH_KEY, poll=5, timeout_min=120)
                # if jdR.get("status") == "Failed":
                #     import json as _json
                #
                #     print(_json.dumps(jdR, ensure_ascii=False, indent=2))
                #     raise SystemExit("❌ 배치 실패(R)")
                # [MODIFIED] 상태 폴링을 L/R 병렬로 처리
                print("\n▶ 상태 폴링 (L/R 병렬)")
                with ThreadPoolExecutor(max_workers=2) as ex:
                    fut_to_ch = {
                        ex.submit(wait_until_done, jobL, SPEECH_KEY, 5, 120): "L",
                        ex.submit(wait_until_done, jobR, SPEECH_KEY, 5, 120): "R",
                    }
                    results = {}
                    for fut in as_completed(fut_to_ch):
                        ch = fut_to_ch[fut]
                        jd = fut.result()  # 예외 발생 시 여기서 즉시 raise
                        print(f" - {ch} 완료: status={jd.get('status')}")
                        if jd.get("status") == "Failed":
                            print(json.dumps(jd, ensure_ascii=False, indent=2))
                            raise SystemExit(f"❌ 배치 실패({ch})")
                        results[ch] = jd

                jdL = results["L"]
                jdR = results["R"]

                print("\n▶ 결과 다운로드/파싱 (L/R)")
                entL = [e for e in list_result_files(jdL, SPEECH_KEY) if e.get("kind") == "Transcription"]
                entR = [e for e in list_result_files(jdR, SPEECH_KEY) if e.get("kind") == "Transcription"]
                if not entL or not entR:
                    raise SystemExit("Transcription JSON 링크가 없습니다. (L/R)")

                # L 저장
                tmpL = out_dir / f"_{base}_tmp_L.json"
                download_text(entL[0]["links"]["contentUrl"], tmpL)
                jsonL = out_dir / f"{base}_batch_L.json"  # [ADDED] 명세화
                tmpL.replace(jsonL)
                print(f"JSON 저장 완료: {jsonL.name}")

                rowsL = parse_transcription_json(jsonL, stereo_mode=False)  # 파일이 이미 단일채널
                csvL = out_dir / f"{base}_batch_L.csv"
                save_csv(rowsL, csvL)
                with open(out_dir / f"{base}_batch_L_rows.json", "w", encoding="utf-8") as jf:
                    json.dump(rowsL, jf, ensure_ascii=False, indent=2)

                # R 저장
                tmpR = out_dir / f"_{base}_tmp_R.json"
                download_text(entR[0]["links"]["contentUrl"], tmpR)
                jsonR = out_dir / f"{base}_batch_R.json"  # [ADDED] 명세화
                tmpR.replace(jsonR)
                print(f"JSON 저장 완료: {jsonR.name}")

                rowsR = parse_transcription_json(jsonR, stereo_mode=False)
                csvR = out_dir / f"{base}_batch_R.csv"
                save_csv(rowsR, csvR)
                with open(out_dir / f"{base}_batch_R_rows.json", "w", encoding="utf-8") as jf:
                    json.dump(rowsR, jf, ensure_ascii=False, indent=2)

                # (옵션) 병합 대화록: TXT만 저장
                if args.dialog:
                    dialog_prefix = out_dir / f"{base}_batch_dialog"
                    left_rows = out_dir / f"{base}_batch_L.json"
                    right_rows = out_dir / f"{base}_batch_R.json"
                    try:
                        # TXT만 생성하도록 호출 (--txt)
                        # cmd = (
                        #     f'python merge_dialog.py '
                        #     f'--left {shlex.quote(str(left_rows))} '
                        #     f'--right {shlex.quote(str(right_rows))} '
                        #     f'--label-left L --label-right R '
                        #     f'--out-prefix {shlex.quote(str(dialog_prefix))} '
                        #     f'--max-pause 0.2 --phrase-split-seconds 5 --fallback-window 2'
                        # )
                        # ✅ 모듈 방식으로 안전하게 실행 (venv/경로/공백 안전)
                        # cmd = [
                        #     sys.executable, "-m", "services.merge_dialog",
                        #     "--left", str(left_rows),
                        #     "--right", str(right_rows),
                        #     "--label-left", "L", "--label-right", "R",
                        #     "--out-prefix", str(dialog_prefix),
                        #     "--max-pause", "0.6",
                        #     # "--phrase-split-seconds", "5", "--fallback-window", "2",
                        # ]
                        cmd = [
                            sys.executable, "-m", "services.merge_LR_dialog",
                            "--left", str(left_rows),
                            "--right", str(right_rows),
                            "--label-left", "L", "--label-right", "R",
                            "--out-prefix", str(dialog_prefix),
                            "--mode", "slice_offset", "--offset-window", "1.2",
                            "--text-field", "itn",  # ← 미지원이면 이 줄 삭제
                            "--txt",
                            "--single-line-if-noswitch",
                            "--max-duration", "5.0",
                            "--word-max-len", "10.0"
                        ]
                        if args.txt:
                            cmd += ["--txt"]
                            if args.txt_no_timestamps:
                                cmd += ["--txt-no-timestamps"]

                            # 중요: shell=False
                        subprocess.check_call(cmd, shell=False)

                        # TXT만 원하면 JSON/CSV는 정리
                        if args.txt:
                            for ext in (".json", ".csv"):
                                p = dialog_prefix.with_suffix(ext)
                                if p.exists():
                                    p.unlink(missing_ok=True)

                        print(
                            "대화 "
                            f"{'TXT ' if args.txt else ''}"
                            "저장 완료: "
                            f"{dialog_prefix.with_suffix('.txt' if args.txt else '.json').name}"
                        )

                    except Exception as e:
                        print("병합 실패(무시):", e)

    print("\n✅ 완료")

