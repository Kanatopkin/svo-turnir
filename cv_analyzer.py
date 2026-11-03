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
    Фиксирует пересечение линии хвата подбородком (оценка по точкам головы и плеч),
    полное разгибание локтей в нижней фазе и угол раскачки корпуса (бедра-плечи).
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

    # Записываем во временный файл, затем перепаковываем в браузерный H.264
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
    notes = []
    bar_y = None
    fully_extended_in_bottom = True
    max_swing_in_rep = 0.0

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

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

                # Определение линии перекладины по положению запястий
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

                # Оценка положения подбородка (интерполяция между носом и линией плеч)
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

                # Оценка угла отклонения корпуса от вертикали (раскачка / киппинг)
                swing_deg = 0.0
                if l_sh[1] > 0 and l_hip[1] > 0:
                    dx = abs(float(l_hip[0] - l_sh[0]))
                    dy = abs(float(l_hip[1] - l_sh[1])) + 1e-5
                    swing_deg = float(np.degrees(np.arctan2(dx, dy)))

                # Логика фаз подтягивания
                if elbow_angle >= elbow_ext_thresh:
                    fully_extended_in_bottom = True
                    if state == "top":
                        state = "hanging"
                        max_swing_in_rep = 0.0

                if state == "hanging" and bar_y is not None and chin_y > 0:
                    max_swing_in_rep = max(max_swing_in_rep, swing_deg)
                    if chin_y <= bar_y and elbow_angle < 115.0:
                        state = "top"
                        time_sec = round(frame_idx / fps, 1)
                        if not fully_extended_in_bottom:
                            invalid_reps += 1
                            notes.append(f"[{time_sec}с] Недоразгиб локтей в висе")
                        elif max_swing_in_rep > swing_thresh_deg:
                            invalid_reps += 1
                            notes.append(f"[{time_sec}с] Раскачка корпуса ({max_swing_in_rep:.0f}°)")
                        else:
                            valid_reps += 1
                        fully_extended_in_bottom = False

                # Отрисовка телеметрии на кадре
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

    # Перекодируем в H.264 для корректной работы в HTML5-плеере Streamlit
    convert_to_browser_mp4(raw_out_path, output_path)
    if os.path.exists(raw_out_path) and raw_out_path != output_path:
        try:
            os.remove(raw_out_path)
        except OSError:
            pass

    summary_note = "; ".join(notes[:5])
    if len(notes) > 5:
        summary_note += f" (и еще {len(notes) - 5})"

    return {
        "valid_reps": valid_reps,
        "invalid_reps": invalid_reps,
        "judge_note": summary_note if summary_note else "Нарушений техники не зафиксировано (AI)",
        "duration_sec": round(total_frames / fps, 1),
        "width": width,
        "height": height,
    }
