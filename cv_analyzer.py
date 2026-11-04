import os
import subprocess
import cv2
import numpy as np


from camera_calibrator import CameraCalibrator
from setup_validator import SetupValidator


def calculate_angle(a, b, c) -> float:
    """Вычисляет плоский угол (в градусах) между тремя точками a, b, c с вершиной в точке b."""
    a = np.array(a[:2], dtype=np.float64)
    b = np.array(b[:2], dtype=np.float64)
    c = np.array(c[:2], dtype=np.float64)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)


def convert_to_browser_mp4(source_path: str, target_path: str) -> None:
    """Перекодирует видео в H.264 (yuv420p) для гарантированного воспроизведения в веб-плеере Streamlit."""
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


def select_main_athlete(results, width: int, height: int, prev_center: np.ndarray | None):
    """
    Выбирает главного спортсмена на турнике среди всех обнаруженных людей в зале.
    Исключает фоновых зрителей, тренеров и людей у тренажеров.
    """
    if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return None, None, None, prev_center

    xy_all = results[0].keypoints.xy.cpu().numpy()
    conf_all = (
        results[0].keypoints.conf.cpu().numpy()
        if results[0].keypoints.conf is not None
        else np.ones(xy_all.shape[:2])
    )
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else None

    best_idx = 0
    best_score = -1e9
    frame_center = np.array([width / 2.0, height / 2.0])

    for i in range(len(xy_all)):
        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = boxes[i][:4]
            area = max(1.0, (x2 - x1) * (y2 - y1))
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        else:
            valid_pts = xy_all[i][xy_all[i][:, 1] > 0]
            if len(valid_pts) == 0:
                continue
            area = float(
                (np.max(valid_pts[:, 0]) - np.min(valid_pts[:, 0]))
                * (np.max(valid_pts[:, 1]) - np.min(valid_pts[:, 1]))
            )
            center = np.mean(valid_pts, axis=0)

        dist_center = np.linalg.norm(center - frame_center)
        dist_prev = np.linalg.norm(center - prev_center) if prev_center is not None else 0.0
        score = area - 1.5 * dist_center - 3.5 * dist_prev
        if score > best_score:
            best_score = score
            best_idx = i

    chosen_xy = xy_all[best_idx].copy()
    chosen_conf = conf_all[best_idx].copy()
    chosen_box = boxes[best_idx][:4].copy() if (boxes is not None and best_idx < len(boxes)) else None
    if chosen_box is not None:
        x1, y1, x2, y2 = chosen_box
        new_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    else:
        new_center = prev_center

    return chosen_xy, chosen_conf, chosen_box, new_center


def analyze_pullup_video(
    video_path: str,
    output_path: str,
    model_name: str = "yolov8n-pose.pt",
    mp_task_path: str = "pose_landmarker_full.task",
    min_hang_sec: float = 0.8,
) -> dict:
    """
    Гибридный анализ техники подтягиваний (YOLO-Pose + MediaPipe Pose + Anchor Grip Lock).
    """
    from ultralytics import YOLO

    yolo_model = YOLO(model_name)

    # 1. Автоматическая калибровка ракурса и перспективы камеры
    calibrator = CameraCalibrator()
    calib_info = calibrator.calibrate_video(video_path, sample_frames=20)

    # 2. Предстартовая валидация кадрирования и геометрии помоста
    validator = SetupValidator()
    setup_info = validator.validate_video(video_path, yolo_model, calib_info=calib_info)

    mp_detector = None
    if os.path.exists(mp_task_path):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=mp_task_path)
            mp_options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                num_poses=1,
            )
            mp_detector = vision.PoseLandmarker.create_from_options(mp_options)
        except Exception as e:
            print("Предупреждение: не удалось загрузить MediaPipe PoseLandmarker:", e)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.7
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dt = 1.0 / fps

    raw_out_path = output_path.replace(".mp4", "_raw_cv.mp4")
    out = None
    for codec in ["mp4v", "avc1", "H264"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        temp_writer = cv2.VideoWriter(raw_out_path, fourcc, fps, (width, height))
        if temp_writer.isOpened():
            out = temp_writer
            break

    if out is None:
        cap.release()
        raise RuntimeError("Не удалось инициализировать VideoWriter для записи видео.")

    # Состояние судейского автомата
    state = "hanging"
    hang_accum_sec = 0.0
    locked_hang_for_rep = 0.0
    pulling_started = False
    attempt_num = 0
    valid_reps = 0
    invalid_reps = 0
    notes = []
    rep_log = []

    # Якоря хвата и калибровки
    bar_y = None
    locked_l_wrist = None
    locked_r_wrist = None
    baseline_torso_len = None
    baseline_hip_width = None
    baseline_thigh_len = None
    baseline_shin_len = None
    baseline_hip_x = None
    prev_center = None

    # Счетчики нарушений текущего повторения (серии кадров и пиковые значения)
    rep_asym_consec = 0
    rep_max_asym_consec = 0
    rep_knees_up_consec = 0
    rep_knees_back_consec = 0
    rep_legs_apart_consec = 0
    rep_max_legs_apart_consec = 0
    rep_kipping_consec = 0
    rep_head_throw_frames = 0
    min_neck_len = 999.0

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        time_sec = round(frame_idx / fps, 1)

        results = yolo_model(frame, verbose=False)
        annotated = frame.copy()

        kpts, confs, main_box, prev_center = select_main_athlete(results, width, height, prev_center)

        elbow_angle = 180.0
        display_el_diff = 0.0

        if kpts is not None and kpts.shape[0] >= 17:
            nose = kpts[0]
            l_ear, r_ear = kpts[3], kpts[4]
            l_sh, r_sh = kpts[5], kpts[6]
            l_el, r_el = kpts[7], kpts[8]
            l_wr, r_wr = kpts[9], kpts[10]
            l_hip, r_hip = kpts[11], kpts[12]
            l_knee, r_knee = kpts[13], kpts[14]
            l_ank, r_ank = kpts[15], kpts[16]

            # Базовая длина корпуса и ширина таза
            torso_len_curr = (
                float(0.5 * ((l_hip[1] - l_sh[1]) + (r_hip[1] - r_sh[1])))
                if (l_sh[1] > 0 and r_sh[1] > 0 and l_hip[1] > 0 and r_hip[1] > 0)
                else (baseline_torso_len or 125.0)
            )
            if torso_len_curr > 40:
                baseline_torso_len = (
                    torso_len_curr
                    if baseline_torso_len is None
                    else 0.95 * baseline_torso_len + 0.05 * torso_len_curr
                )
            t_len = baseline_torso_len or 125.0

            hip_width_curr = (
                abs(float(l_hip[0] - r_hip[0]))
                if (l_hip[1] > 0 and r_hip[1] > 0)
                else (baseline_hip_width or 0.26 * t_len)
            )
            if hip_width_curr > 15:
                baseline_hip_width = (
                    hip_width_curr
                    if baseline_hip_width is None
                    else 0.95 * baseline_hip_width + 0.05 * hip_width_curr
                )
            h_width = max(baseline_hip_width or 0.26 * t_len, 0.20 * t_len)

            # 2. ANCHOR GRIP LOCK (фиксация запястий на перекладине)
            if l_wr[1] > 0 and (confs is None or confs[9] >= 0.45):
                locked_l_wrist = l_wr.copy() if locked_l_wrist is None else 0.88 * locked_l_wrist + 0.12 * l_wr
            elif locked_l_wrist is not None:
                l_wr = locked_l_wrist.copy()
                kpts[9] = l_wr

            if r_wr[1] > 0 and (confs is None or confs[10] >= 0.45):
                locked_r_wrist = r_wr.copy() if locked_r_wrist is None else 0.88 * locked_r_wrist + 0.12 * r_wr
            elif locked_r_wrist is not None:
                r_wr = locked_r_wrist.copy()
                kpts[10] = r_wr

            if locked_l_wrist is not None and locked_r_wrist is not None:
                wr_y_mean = float(0.5 * (locked_l_wrist[1] + locked_r_wrist[1]))
                if bar_y is None or frame_idx <= 45:
                    bar_y = wr_y_mean if bar_y is None else 0.9 * bar_y + 0.1 * wr_y_mean

            sh_y_mean = float(0.5 * (l_sh[1] + r_sh[1]))
            in_low_hang_zone = bar_y is not None and (sh_y_mean - bar_y) > 0.30 * t_len

            # 3. MediaPipe на кропе спортсмена
            mp_el_diff = 0.0
            mp_legs_apart = False
            if mp_detector is not None and main_box is not None:
                bx1, by1, bx2, by2 = [int(v) for v in main_box]
                cx1, cy1 = max(0, bx1 - 30), max(0, by1 - 30)
                cx2, cy2 = min(width, bx2 + 30), min(height, by2 + 30)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    try:
                        import mediapipe as mp
                        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                        mp_res = mp_detector.detect(mp_img)
                        if mp_res.pose_landmarks and len(mp_res.pose_landmarks) > 0:
                            lm = mp_res.pose_landmarks[0]
                            cw, ch = crop.shape[1], crop.shape[0]
                            if lm[15].visibility > 0.35 and lm[16].visibility > 0.35:
                                mp_la = calculate_angle([lm[11].x * cw, lm[11].y * ch], [lm[13].x * cw, lm[13].y * ch], [lm[15].x * cw, lm[15].y * ch])
                                mp_ra = calculate_angle([lm[12].x * cw, lm[12].y * ch], [lm[14].x * cw, lm[14].y * ch], [lm[16].x * cw, lm[16].y * ch])
                                mp_mean = 0.5 * (mp_la + mp_ra)
                                if 35.0 < mp_mean < 145.0:
                                    mp_el_diff = abs(mp_la - mp_ra)
                            mp_hip_w = abs(lm[23].x - lm[24].x) * cw
                            if mp_hip_w > 10:
                                if lm[27].visibility > 0.35 and lm[28].visibility > 0.35:
                                    if abs(lm[27].x - lm[28].x) * cw > 1.18 * mp_hip_w:
                                        mp_legs_apart = True
                                if lm[25].visibility > 0.35 and lm[26].visibility > 0.35:
                                    if abs(lm[25].x - lm[26].x) * cw > 1.18 * mp_hip_w:
                                        mp_legs_apart = True
                    except Exception:
                        pass

            # Углы локтей и перекос плеч
            y_la = calculate_angle(l_sh, l_el, l_wr)
            y_ra = calculate_angle(r_sh, r_el, r_wr)
            elbow_angle = 0.5 * (y_la + y_ra)
            el_diff = max(abs(y_la - y_ra), mp_el_diff)
            display_el_diff = el_diff

            sh_dx = abs(float(l_sh[0] - r_sh[0])) + 1e-5
            sh_dy = abs(float(l_sh[1] - r_sh[1]))
            shoulder_tilt_deg = float(np.degrees(np.arctan2(sh_dy, sh_dx)))

            # Углы коленей и бедер
            knee_angles = []
            if l_hip[1] > 0 and l_knee[1] > 0 and l_ank[1] > 0:
                knee_angles.append(calculate_angle(l_hip, l_knee, l_ank))
            if r_hip[1] > 0 and r_knee[1] > 0 and r_ank[1] > 0:
                knee_angles.append(calculate_angle(r_hip, r_knee, r_ank))
            knee_angle = float(np.min(knee_angles)) if knee_angles else 180.0

            hip_angles = []
            if l_sh[1] > 0 and l_hip[1] > 0 and l_knee[1] > 0:
                hip_angles.append(calculate_angle(l_sh, l_hip, l_knee))
            if r_sh[1] > 0 and r_hip[1] > 0 and r_knee[1] > 0:
                hip_angles.append(calculate_angle(r_sh, r_hip, r_knee))
            hip_angle = float(np.min(hip_angles)) if hip_angles else 180.0

            thigh_len_curr = float(0.5 * ((l_knee[1] - l_hip[1]) + (r_knee[1] - r_hip[1])))
            shin_len_curr = float(0.5 * ((l_ank[1] - l_knee[1]) + (r_ank[1] - r_knee[1])))

            # Проверка ошибки «Ноги не вместе (разведение ног)»
            ank_valid = (
                l_ank[1] > 0
                and r_ank[1] > 0
                and (confs is None or (confs[15] > 0.25 and confs[16] > 0.25))
            )
            knee_valid = (
                l_knee[1] > 0
                and r_knee[1] > 0
                and (confs is None or (confs[13] > 0.25 and confs[14] > 0.25))
            )
            ank_spread = abs(float(l_ank[0] - r_ank[0])) if ank_valid else 0.0
            knee_spread = abs(float(l_knee[0] - r_knee[0])) if knee_valid else 0.0

            is_legs_apart = (
                ank_spread > max(1.15 * h_width, 0.30 * t_len)
                or knee_spread > max(1.15 * h_width, 0.29 * t_len)
                or mp_legs_apart
            )

            # Раскачка (киппинг)
            sh_center = 0.5 * (l_sh + r_sh)
            hip_center = 0.5 * (l_hip + r_hip)
            ank_center = 0.5 * (l_ank + r_ank)

            torso_swing_deg = float(
                np.degrees(np.arctan2(abs(hip_center[0] - sh_center[0]), abs(hip_center[1] - sh_center[1]) + 1e-5))
            )
            leg_swing_deg = float(
                np.degrees(np.arctan2(abs(ank_center[0] - hip_center[0]), abs(ank_center[1] - hip_center[1]) + 1e-5))
            )
            kipping_deg = max(torso_swing_deg, 0.8 * leg_swing_deg)

            # Положение головы и подбородка
            ear_y = float(0.5 * (l_ear[1] + r_ear[1])) if (l_ear[1] > 0 and r_ear[1] > 0) else float(nose[1])
            chin_y = float(nose[1] + 0.35 * (sh_y_mean - nose[1]))
            is_head_thrown = (float(nose[1]) >= ear_y - 2.0) and (elbow_angle < 105.0)
            neck_len = sh_y_mean - ear_y

            # 4. Определение спокойного нижнего виса на прямых руках
            is_in_bottom = (elbow_angle >= 146.0) and in_low_hang_zone

            if is_in_bottom:
                if state in ("top", "descending"):
                    state = "hanging"
                    hang_accum_sec = 0.0
                    pulling_started = False
                    rep_asym_consec = 0
                    rep_max_asym_consec = 0
                    rep_knees_up_consec = 0
                    rep_knees_back_consec = 0
                    rep_legs_apart_consec = 0
                    rep_max_legs_apart_consec = 0
                    rep_kipping_consec = 0
                    rep_head_throw_frames = 0
                    min_neck_len = 999.0

                hang_accum_sec += dt
                locked_hang_for_rep = max(locked_hang_for_rep, hang_accum_sec)

                # Калибровка пропорций тела в спокойном висе
                if thigh_len_curr > 25:
                    baseline_thigh_len = (
                        thigh_len_curr
                        if baseline_thigh_len is None
                        else 0.95 * baseline_thigh_len + 0.05 * thigh_len_curr
                    )
                if shin_len_curr > 25:
                    baseline_shin_len = (
                        shin_len_curr
                        if baseline_shin_len is None
                        else 0.95 * baseline_shin_len + 0.05 * shin_len_curr
                    )
                baseline_hip_x = (
                    float(hip_center[0])
                    if baseline_hip_x is None
                    else 0.92 * baseline_hip_x + 0.08 * float(hip_center[0])
                )
            else:
                if not pulling_started and (
                    elbow_angle < 140.0
                    or (bar_y is not None and (sh_y_mean - bar_y) < 0.28 * t_len)
                ):
                    pulling_started = True
                    locked_hang_for_rep = hang_accum_sec
                    hang_accum_sec = 0.0

                if pulling_started:
                    if (35.0 < elbow_angle < 140.0) and (
                        el_diff > 18.0 or shoulder_tilt_deg > 9.5
                    ):
                        rep_asym_consec += 1
                        rep_max_asym_consec = max(rep_max_asym_consec, rep_asym_consec)
                    else:
                        rep_asym_consec = 0

                    if hip_angle < 155.0 or (
                        baseline_thigh_len and thigh_len_curr / baseline_thigh_len < 0.75
                    ):
                        rep_knees_up_consec += 1

                    if (
                        knee_angle < 148.0
                        or (baseline_shin_len and shin_len_curr / baseline_shin_len < 0.72)
                    ) and hip_angle >= 155.0:
                        rep_knees_back_consec += 1

                    # НАКОПЛЕНИЕ ОШИБКИ РАЗВЕДЕНИЯ НОГ В ПРОЦЕССЕ ТЯГИ
                    if is_legs_apart:
                        rep_legs_apart_consec += 1
                        rep_max_legs_apart_consec = max(rep_max_legs_apart_consec, rep_legs_apart_consec)

                    if kipping_deg > 8.0:
                        rep_kipping_consec += 1

                    if is_head_thrown:
                        rep_head_throw_frames += 1
                    min_neck_len = min(min_neck_len, neck_len)

            if state == "hanging" and bar_y is not None and pulling_started:
                if (float(nose[1]) <= bar_y + 35.0) and (elbow_angle < 110.0):
                    state = "top"
                    attempt_num += 1

            if state == "top":
                if elbow_angle > 115.0 or (
                    bar_y is not None and (sh_y_mean - bar_y) > 0.40 * t_len
                ):
                    errs = []
                    if locked_hang_for_rep < min_hang_sec and attempt_num > 1:
                        errs.append(f"нет фиксации виса 0.8с ({locked_hang_for_rep:.2f}с)")

                    if rep_knees_up_consec >= 5:
                        errs.append("поднятие колен вверх (складывание в тазобедренном суставе)")
                    elif rep_knees_back_consec >= 5:
                        errs.append("сгибание ног в коленях назад")
                    elif rep_max_legs_apart_consec >= 3:
                        errs.append("ноги не вместе (разведение ног)")
                    elif rep_head_throw_frames >= 22 and min_neck_len < 32.0:
                        errs.append("запрокидывание головы назад и вытягивание подбородком")
                    elif rep_kipping_consec >= 3 and rep_max_asym_consec <= 12:
                        errs.append("рывок / раскачка корпуса (киппинг)")
                    elif rep_max_asym_consec >= 3:
                        errs.append("поочередное сгибание рук / перекос плеч")
                    elif rep_kipping_consec >= 3:
                        errs.append("рывок / раскачка корпуса (киппинг)")

                    if errs:
                        invalid_reps += 1
                        reason_str = ", ".join(errs)
                        notes.append(f"Попытка #{attempt_num} [{time_sec}с]: НЕ ЗАЧТЕНО — {reason_str}")
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
                            "reason": f"Эталонное повторение (вис {locked_hang_for_rep:.2f}с)",
                        })

                    state = "descending"
                    pulling_started = False
                    hang_accum_sec = 0.0
                    locked_hang_for_rep = 0.0
                    rep_legs_apart_consec = 0
                    rep_max_legs_apart_consec = 0

            # Отрисовка бокса и скелета спортсмена
            if main_box is not None:
                bx1, by1, bx2, by2 = [int(v) for v in main_box]
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 80, 0), 2)

            skeleton_links = [
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
            ]
            for i_a, i_b in skeleton_links:
                pt_a, pt_b = kpts[i_a], kpts[i_b]
                if pt_a[1] > 0 and pt_b[1] > 0:
                    cv2.line(annotated, (int(pt_a[0]), int(pt_a[1])), (int(pt_b[0]), int(pt_b[1])), (0, 200, 255), 2)
            for idx_pt in [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                pt = kpts[idx_pt]
                if pt[1] > 0:
                    cv2.circle(annotated, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

            # Визуальная подсветка ошибки разведения ног на кадре
            if is_legs_apart:
                if ank_valid:
                    cv2.line(annotated, (int(l_ank[0]), int(l_ank[1])), (int(r_ank[0]), int(r_ank[1])), (0, 0, 255), 3)
                if knee_valid:
                    cv2.line(annotated, (int(l_knee[0]), int(l_knee[1])), (int(r_knee[0]), int(r_knee[1])), (0, 0, 255), 3)

            # Отрисовка якорей хвата Anchor Grip Lock
            if locked_l_wrist is not None:
                cv2.circle(annotated, (int(locked_l_wrist[0]), int(locked_l_wrist[1])), 9, (0, 0, 255), 2)
            if locked_r_wrist is not None:
                cv2.circle(annotated, (int(locked_r_wrist[0]), int(locked_r_wrist[1])), 9, (0, 0, 255), 2)

        # Линия перекладины
        if bar_y is not None:
            cv2.line(annotated, (0, int(bar_y)), (width, int(bar_y)), (0, 255, 255), 2)
            cv2.putText(annotated, "Bar Level", (15, int(bar_y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Верхний левый угол: Репс, Ошибки, Угол локтей, Асимметрия
        cv2.rectangle(annotated, (10, 10), (480, 52), (0, 0, 0), -1)
        hud_left = f"Reps: {valid_reps} | Err: {invalid_reps} | El: {int(elbow_angle)}d | dEl: {int(display_el_diff)}d"
        cv2.putText(annotated, hud_left, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # Верхний правый угол: Таймер спокойного виса >= 0.8с
        current_hang = hang_accum_sec if (state == "hanging" and not pulling_started) else locked_hang_for_rep
        hang_ok = current_hang >= min_hang_sec
        hang_color = (0, 255, 0) if hang_ok else (0, 140, 255)
        hang_status = "OK" if hang_ok else "WAIT"
        hang_text = f"Hang: {current_hang:.2f}s / {min_hang_sec:.1f}s [{hang_status}]"

        (tw, _), _ = cv2.getTextSize(hang_text, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
        box_x1 = max(10, width - tw - 34)
        cv2.rectangle(annotated, (box_x1, 10), (width - 10, 52), (0, 0, 0), -1)
        cv2.putText(annotated, hang_text, (box_x1 + 12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, hang_color, 2)

        # Нижний левый угол: Индикатор калибровки ракурса камеры
        calib_roll = calib_info.get("roll_angle_deg", 0.0)
        calib_pitch = calib_info.get("pitch_tilt_deg", 0.0)
        calib_hud = f"Cam: Roll {calib_roll:+.1f}d | Pitch {calib_pitch:+.1f}d"
        cv2.rectangle(annotated, (10, height - 42), (320, height - 10), (0, 0, 0), -1)
        cv2.putText(annotated, calib_hud, (18, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

        out.write(annotated)

    cap.release()
    out.release()

    # Перекодирование в гарантированный H.264
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
        "calib_info": calib_info,
        "setup_info": setup_info,
    }
