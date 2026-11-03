import os
import subprocess
import cv2
import numpy as np


def calculate_angle(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)


def convert_to_browser_mp4(source_path: str, target_path: str) -> None:
    """Перекодирует видео в H.264 (yuv420p) для гарантированного воспроизведения в браузере Streamlit."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        temp_h264 = target_path.replace(".mp4", "_h264_tmp.mp4")
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i",
                source_path,
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-movflags",
                "+faststart",
                temp_h264,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        os.replace(temp_h264, target_path)
    except Exception as e:
        print("Предупреждение перекодирования FFmpeg:", e)
        if source_path != target_path and os.path.exists(source_path):
            os.replace(source_path, target_path)


def analyze_pullup_video(
    video_path: str,
    output_path: str,
    model_name: str = "yolov8n-pose.pt",
    elbow_ext_thresh: float = 155.0,
    swing_thresh_deg: float = 20.0,
) -> dict:
    """
    Локальный анализатор техники подтягиваний на базе Ultralytics YOLO-Pose.
    Фиксирует каждую попытку, зачетные повторения и детальный список ошибок с таймкодами:
    - Недоразгиб локтей в нижнем висе (с указанием максимального угла разгибания);
    - Маятниковая раскачка / киппинг корпуса (с указанием угла отклонения от вертикали);
    - Недотяг подбородка до уровня перекладины при попытке подъема.
    """
    from ultralytics import YOLO

    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw_out_path = output_path.replace(".mp4", "_raw_cv.mp4")
    out = None
    for codec in ["avc1", "H264", "mp4v"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        temp_writer = cv2.VideoWriter(raw_out_path, fourcc, fps, (width, height))
        if temp_writer.isOpened():
            out = temp_writer
            break

    if out is None:
        cap.release()
        raise RuntimeError("Не удалось инициализировать VideoWriter для записи видео.")

    state = "hanging"
    valid_reps = 0
    invalid_reps = 0
    attempt_num = 0
    notes = []
    rep_log = []

    bar_y = None
    fully_extended_in_bottom = True
    max_bottom_elbow_angle = 180.0
    max_swing_in_rep = 0.0
    pulling_started = False
    min_chin_y_in_pull = 99999.0
    min_elbow_in_pull = 180.0

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        time_sec = round(frame_idx / fps, 1)

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            if kpts.shape[0] >= 13:
                nose = kpts[0]
                l_sh, r_sh = kpts[5], kpts[6]
                l_el, r_el = kpts[7], kpts[8]
                l_wr, r_wr = kpts[9], kpts[10]
                l_hip, r_hip = kpts[11], kpts[12]

                # Определение линии перекладины по запястьям
                valid_wr_y = [pt[1] for pt in (l_wr, r_wr) if pt[1] > 0]
                if valid_wr_y:
                    curr_wr_y = float(np.mean(valid_wr_y))
                    bar_y = curr_wr_y if bar_y is None else 0.9 * bar_y + 0.1 * curr_wr_y

                # Угол в локтевом суставе
                angles = []
                if l_sh[1] > 0 and l_el[1] > 0 and l_wr[1] > 0:
                    angles.append(calculate_angle(l_sh, l_el, l_wr))
                if r_sh[1] > 0 and r_el[1] > 0 and r_wr[1] > 0:
                    angles.append(calculate_angle(r_sh, r_el, r_wr))
                elbow_angle = float(np.mean(angles)) if angles else 180.0

                # Оценка положения подбородка
                sh_y = (
                    float(np.mean([pt[1] for pt in (l_sh, r_sh) if pt[1] > 0]))
                    if (l_sh[1] > 0 or r_sh[1] > 0)
                    else 0.0
                )
                chin_y = (
                    float(nose[1] + 0.35 * (sh_y - nose[1]))
                    if (nose[1] > 0 and sh_y > 0)
                    else float(nose[1])
                )

                # Угол отклонения корпуса (раскачка / киппинг)
                swing_deg = 0.0
                if l_sh[1] > 0 and l_hip[1] > 0:
                    dx = abs(float(l_hip[0] - l_sh[0]))
                    dy = abs(float(l_hip[1] - l_sh[1])) + 1e-5
                    swing_deg = float(np.degrees(np.arctan2(dx, dy)))

                # Фиксация нижней фазы виса
                if elbow_angle >= elbow_ext_thresh:
                    # Если участник сгибал руки (< 115°), но опустился вниз, так и не пересекая гриф подбородком
                    if state == "hanging" and pulling_started and min_elbow_in_pull < 115.0 and bar_y is not None:
                        if min_chin_y_in_pull > bar_y:
                            attempt_num += 1
                            invalid_reps += 1
                            err_text = f"Попытка #{attempt_num} [{time_sec}с]: Недотяг — подбородок ниже перекладины"
                            notes.append(err_text)
                            rep_log.append({
                                "attempt": attempt_num,
                                "time_sec": time_sec,
                                "status": "Ошибка",
                                "reason": "Недотяг (подбородок не пересек линию перекладины)",
                            })
                    pulling_started = False
                    min_chin_y_in_pull = 99999.0
                    min_elbow_in_pull = 180.0
                    fully_extended_in_bottom = True
                    max_bottom_elbow_angle = max(max_bottom_elbow_angle, elbow_angle)
                    if state == "top":
                        state = "hanging"
                        max_swing_in_rep = 0.0
                else:
                    if state == "hanging":
                        max_bottom_elbow_angle = max(max_bottom_elbow_angle, elbow_angle)
                        if elbow_angle < 135.0:
                            pulling_started = True
                            min_elbow_in_pull = min(min_elbow_in_pull, elbow_angle)
                            if chin_y > 0:
                                min_chin_y_in_pull = min(min_chin_y_in_pull, chin_y)

                # Проверка верхней мертвой точки
                if state == "hanging" and bar_y is not None and chin_y > 0:
                    max_swing_in_rep = max(max_swing_in_rep, swing_deg)
                    if chin_y <= bar_y and elbow_angle < 115.0:
                        state = "top"
                        attempt_num += 1
                        pulling_started = False
                        min_chin_y_in_pull = 99999.0
                        min_elbow_in_pull = 180.0

                        rep_errors = []
                        if not fully_extended_in_bottom:
                            rep_errors.append(
                                f"недоразгиб локтей в висе ({int(max_bottom_elbow_angle)}° при норме ≥{int(elbow_ext_thresh)}°)"
                            )
                        if max_swing_in_rep > swing_thresh_deg:
                            rep_errors.append(
                                f"раскачка корпуса / киппинг ({int(max_swing_in_rep)}° при норме ≤{int(swing_thresh_deg)}°)"
                            )

                        if rep_errors:
                            invalid_reps += 1
                            reason_str = ", ".join(rep_errors)
                            err_msg = f"Попытка #{attempt_num} [{time_sec}с]: НЕ ЗАЧТЕНО — {reason_str}"
                            notes.append(err_msg)
                            rep_log.append({
                                "attempt": attempt_num,
                                "time_sec": time_sec,
                                "status": "Ошибка",
                                "reason": reason_str.capitalize(),
                            })
                        else:
                            valid_reps += 1
                            rep_log.append({
                                "attempt": attempt_num,
                                "time_sec": time_sec,
                                "status": "Зачет",
                                "reason": "Чистое повторение",
                            })

                        fully_extended_in_bottom = False
                        max_bottom_elbow_angle = elbow_angle
                        max_swing_in_rep = 0.0

                # Отрисовка телеметрии на видео
                if bar_y is not None:
                    cv2.line(annotated, (0, int(bar_y)), (width, int(bar_y)), (0, 255, 255), 2)
                cv2.putText(
                    annotated,
                    f"Reps: {valid_reps} | Err: {invalid_reps} | Elbow: {int(elbow_angle)} deg",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

        out.write(annotated)

    cap.release()
    out.release()

    convert_to_browser_mp4(raw_out_path, output_path)
    if os.path.exists(raw_out_path) and raw_out_path != output_path:
        try:
            os.remove(raw_out_path)
        except OSError:
            pass

    if notes:
        summary_note = (
            f"Засчитано: {valid_reps} (всего попыток: {attempt_num}, ошибок: {invalid_reps}).\n"
            + "\n".join(notes)
        )
    else:
        summary_note = (
            f"Засчитано чистых повторений: {valid_reps} (всего попыток: {attempt_num}). "
            "Нарушений техники не зафиксировано (AI)."
        )

    return {
        "total_attempts": attempt_num,
        "valid_reps": valid_reps,
        "invalid_reps": invalid_reps,
        "errors_list": notes,
        "rep_log": rep_log,
        "judge_note": summary_note,
        "duration_sec": round(total_frames / fps, 1),
        "width": width,
        "height": height,
    }
