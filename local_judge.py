import os
import json
import streamlit as st
import pandas as pd
from db import get_db, calculate_team_score
from gdrive import download_video_to_local
from cv_analyzer import analyze_pullup_video, convert_to_browser_mp4

st.set_page_config(
    page_title="Локальный пульт ГСК — Судейство и AI-видеоанализ",
    page_icon="⚖️",
    layout="wide",
)

VIDEO_CACHE_DIR = os.path.join(os.getcwd(), "local_video_cache")
os.makedirs(VIDEO_CACHE_DIR, exist_ok=True)

db = get_db()


def render_video_file(file_path: str):
    """Безопасная передача локального видеофайла в плеер Streamlit через байтовый поток."""
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "rb") as vf:
            st.video(vf.read())
    else:
        st.warning("Файл видео пуст или поврежден.")


def load_cached_report(report_path: str) -> dict | None:
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


st.sidebar.title("⚖️ Пульт ГСК (Локальный)")
mode = st.sidebar.radio(
    "Раздел судейской коллегии",
    [
        "Проверка видео и AI-судейство",
        "Синхронизация видео с Google Drive",
        "Управление финальной сеткой (Топ-8)",
        "Экспорт официального протокола (Excel)",
    ],
)

st.sidebar.divider()
if st.sidebar.button("🛑 Выключить сервер ГСК"):
    st.sidebar.warning("Сервер остановлен. Можете закрыть вкладку браузера.")
    os._exit(0)

if mode == "Проверка видео и AI-судейство":
    st.title("Локальное рабочее место судьи (ГСК)")
    st.caption(
        "Видео скачивается с Google Drive в локальный кэш компьютера судьи для мгновенного воспроизведения и локального анализа через YOLO-Pose."
    )

    teams = db.table("teams").select("*").order("id").execute().data or []
    if not teams:
        st.info("Зарегистрированных школьных команд пока нет.")
    else:
        team_options = {
            f"{t['school_name']} ({t['municipality']}) — [{t['status']}]": t for t in teams
        }
        selected_label = st.selectbox("Выберите команду для проверки", list(team_options.keys()))
        team = team_options[selected_label]

        roster = (
            db.table("participants")
            .select("*")
            .eq("team_id", team["id"])
            .order("id")
            .execute()
            .data
            or []
        )
        score, scoring_ids = calculate_team_score(roster)

        c1, c2, c3 = st.columns([2, 1, 1])
        c1.metric("Командный результат (Сумма Топ-10)", score)
        c2.metric("Статус публикации", "Опубликована" if team["is_published"] else "Скрыта")
        with c3:
            pub_label = (
                "Снять с публикации" if team["is_published"] else "Опубликовать в протоколе"
            )
            if st.button(pub_label, use_container_width=True):
                db.table("teams").update({"is_published": not team["is_published"]}).eq(
                    "id", team["id"]
                ).execute()
                st.rerun()

        st.divider()

        for p in roster:
            local_raw_path = os.path.join(
                VIDEO_CACHE_DIR, f"team_{team['id']}_p_{p['id']}_raw.mp4"
            )
            local_ai_path = os.path.join(
                VIDEO_CACHE_DIR, f"team_{team['id']}_p_{p['id']}_yolo.mp4"
            )
            local_report_path = os.path.join(
                VIDEO_CACHE_DIR, f"team_{team['id']}_p_{p['id']}_report.json"
            )

            cached_report = load_cached_report(local_report_path)
            ai_badge = (
                f" | 🤖 AI-разбор: {cached_report['valid_reps']} зач. / {cached_report['invalid_reps']} ош."
                if cached_report
                else ""
            )

            with st.expander(
                f"{p['full_name']} ({p['class_name']} кл.) — Статус: {p['video_status']} | Зачтено ГСК: {p.get('verified_reps') or 0}{ai_badge}",
                expanded=(p["video_status"] == "uploaded"),
            ):
                col_video, col_judge = st.columns([3, 2])

                with col_video:
                    if not p.get("video_drive_id"):
                        st.warning("Участник еще не загрузил видео.")
                    else:
                        b_dl, b_ai = st.columns(2)
                        if b_dl.button("📥 Скачать с Google Drive в кэш", key=f"dl_{p['id']}"):
                            with st.spinner("Скачивание оригинала с Google Drive..."):
                                if os.path.exists(local_raw_path):
                                    os.remove(local_raw_path)
                                download_video_to_local(p["video_drive_id"], local_raw_path)
                                convert_to_browser_mp4(local_raw_path, local_raw_path)
                                st.success("Видео сохранено локально!")
                                st.rerun()

                        if b_ai.button("🤖 Запустить локальный YOLO-Pose разбор", key=f"ai_{p['id']}"):
                            if not os.path.exists(local_raw_path):
                                with st.spinner("Скачивание оригинала с Google Drive..."):
                                    download_video_to_local(p["video_drive_id"], local_raw_path)
                                    convert_to_browser_mp4(local_raw_path, local_raw_path)
                            with st.spinner("Выполняется локальный нейросетевой анализ техники..."):
                                if os.path.exists(local_ai_path):
                                    os.remove(local_ai_path)
                                report = analyze_pullup_video(local_raw_path, local_ai_path)
                                with open(local_report_path, "w", encoding="utf-8") as rf:
                                    json.dump(report, rf, ensure_ascii=False, indent=2)

                                # Напрямую обновляем ключи виджетов ввода, чтобы значения сразу появились на экране
                                st.session_state[f"reps_in_{p['id']}"] = int(report["valid_reps"])
                                st.session_state[f"note_in_{p['id']}"] = str(report["judge_note"])
                                st.rerun()

                        if os.path.exists(local_ai_path) and os.path.exists(local_raw_path):
                            tab_ai, tab_raw = st.tabs(
                                ["🤖 Видео с разметкой YOLO-Pose", "📹 Оригинал видео"]
                            )
                            with tab_ai:
                                render_video_file(local_ai_path)
                            with tab_raw:
                                render_video_file(local_raw_path)
                        elif os.path.exists(local_ai_path):
                            st.caption("Видео с наложенной разметкой YOLO-Pose:")
                            render_video_file(local_ai_path)
                        elif os.path.exists(local_raw_path):
                            st.caption("Оригинальное видео (из локального кэша):")
                            render_video_file(local_raw_path)
                        elif p.get("video_url"):
                            st.markdown(
                                f"[🔗 Открыть видео в облаке Google Drive]({p['video_url']})"
                            )

                with col_judge:
                    # Отображение сводки компьютерного зрения YOLO-Pose
                    if cached_report:
                        with st.container(border=True):
                            st.markdown("**📊 Результаты анализа ИИ (YOLO-Pose)**")
                            mc1, mc2, mc3 = st.columns(3)
                            mc1.metric("Попыток", cached_report.get("total_attempts", 0))
                            mc2.metric("✅ Засчитано", cached_report.get("valid_reps", 0))
                            mc3.metric("❌ Ошибок", cached_report.get("invalid_reps", 0))

                            errors_list = cached_report.get("errors_list", [])
                            if errors_list:
                                st.markdown("**⚠️ Зафиксированные ошибки техники:**")
                                for err_item in errors_list:
                                    st.markdown(f"- {err_item}")
                            else:
                                st.success("Нарушений техники не обнаружено.")

                            rep_log = cached_report.get("rep_log", [])
                            if rep_log:
                                with st.expander("Повторения по таймкодам"):
                                    for r_item in rep_log:
                                        icon = "✅" if r_item["status"] == "Зачет" else "❌"
                                        st.caption(
                                            f"{icon} **#{r_item['attempt']}** ({r_item['time_sec']} сек) — {r_item['reason']}"
                                        )

                            if st.button(
                                "⤵️ Перенести данные ИИ в протокол судьи",
                                key=f"fill_ai_{p['id']}",
                                use_container_width=True,
                            ):
                                st.session_state[f"reps_in_{p['id']}"] = int(
                                    cached_report.get("valid_reps", 0)
                                )
                                st.session_state[f"note_in_{p['id']}"] = str(
                                    cached_report.get("judge_note", "")
                                )
                                st.rerun()

                    # Инициализация значений виджетов при первом открытии
                    if f"reps_in_{p['id']}" not in st.session_state:
                        if p.get("verified_reps") is not None:
                            st.session_state[f"reps_in_{p['id']}"] = int(p["verified_reps"])
                        elif cached_report:
                            st.session_state[f"reps_in_{p['id']}"] = int(
                                cached_report.get("valid_reps", 0)
                            )
                        else:
                            st.session_state[f"reps_in_{p['id']}"] = 0

                    if f"note_in_{p['id']}" not in st.session_state:
                        if p.get("judge_note"):
                            st.session_state[f"note_in_{p['id']}"] = str(p["judge_note"])
                        elif cached_report:
                            st.session_state[f"note_in_{p['id']}"] = str(
                                cached_report.get("judge_note", "")
                            )
                        else:
                            st.session_state[f"note_in_{p['id']}"] = ""

                    reps_val = st.number_input(
                        "Засчитанные чистые повторения",
                        min_value=0,
                        max_value=200,
                        key=f"reps_in_{p['id']}",
                    )
                    note_val = st.text_area(
                        "Судейский комментарий / описание ошибок и таймкоды",
                        height=130,
                        key=f"note_in_{p['id']}",
                    )

                    ba1, ba2 = st.columns(2)
                    if ba1.button("✅ Зачесть результат", key=f"app_{p['id']}", type="primary"):
                        db.table("participants").update(
                            {
                                "verified_reps": int(reps_val),
                                "video_status": "approved",
                                "judge_note": note_val.strip() or None,
                            }
                        ).eq("id", p["id"]).execute()
                        st.success("Результат утвержден в БД!")
                        st.rerun()

                    if ba2.button("❌ Отклонить видео", key=f"rej_{p['id']}"):
                        db.table("participants").update(
                            {
                                "verified_reps": 0,
                                "video_status": "rejected",
                                "judge_note": note_val.strip() or "Отклонено решением ГСК",
                            }
                        ).eq("id", p["id"]).execute()
                        st.warning("Попытка отклонена.")
                        st.rerun()

elif mode == "Синхронизация видео с Google Drive":
    st.title("Пакетное скачивание всех видеозаписей на компьютер судьи")
    st.write(f"Папка локального кэша: `{VIDEO_CACHE_DIR}`")
    all_parts = (
        db.table("participants")
        .select("*")
        .neq("video_status", "missing")
        .execute()
        .data
        or []
    )
    st.info(f"Всего участников с загруженными видео в базе: {len(all_parts)}")

    if st.button("⬇️ Скачать все недостающие видео с Google Drive"):
        prog = st.progress(0)
        for idx, p in enumerate(all_parts):
            if p.get("video_drive_id"):
                dest = os.path.join(VIDEO_CACHE_DIR, f"team_{p['team_id']}_p_{p['id']}_raw.mp4")
                if not os.path.exists(dest):
                    download_video_to_local(p["video_drive_id"], dest)
                    convert_to_browser_mp4(dest, dest)
            prog.progress((idx + 1) / max(len(all_parts), 1))
        st.success("Все видеозаписи синхронизированы для офлайн-просмотра и нейросетевой проверки!")

elif mode == "Управление финальной сеткой (Топ-8)":
    st.title("Формирование и ведение финальной сетки плей-офф")

    if st.button("🔄 Сформировать 1/4 финала по Топ-8 участникам опубликованных команд"):
        pub_teams = {
            t["id"]: t
            for t in (
                db.table("teams").select("*").eq("is_published", True).execute().data or []
            )
        }
        all_p = (
            db.table("participants").select("*").eq("video_status", "approved").execute().data
            or []
        )
        qualifiers = [p for p in all_p if p["team_id"] in pub_teams]
        qualifiers.sort(key=lambda x: x.get("verified_reps") or 0, reverse=True)
        top8 = qualifiers[:8]

        if len(top8) < 8:
            st.error(
                f"Для генерации сетки нужно минимум 8 участников с утвержденным результатом (сейчас: {len(top8)})."
            )
        else:
            db.table("final_matches").delete().neq("id", 0).execute()
            pairs = [(0, 7), (1, 6), (2, 5), (3, 4)]
            for idx, (a, b) in enumerate(pairs, start=1):
                db.table("final_matches").insert(
                    {
                        "round": "quarterfinal",
                        "match_no": idx,
                        "participant_a_id": top8[a]["id"],
                        "participant_b_id": top8[b]["id"],
                    }
                ).execute()
            st.success("Четвертьфинальные пары (1-8, 2-7, 3-6, 4-5) успешно сформированы!")
            st.rerun()

    matches = db.table("final_matches").select("*").order("id").execute().data or []
    p_map = {p["id"]: p for p in (db.table("participants").select("*").execute().data or [])}

    for m in matches:
        pa = p_map.get(m.get("participant_a_id"))
        pb = p_map.get(m.get("participant_b_id"))
        if pa and pb:
            with st.container(border=True):
                st.markdown(f"**Раунд: {m['round'].upper()} | Матч #{m['match_no']}**")
                c1, c2 = st.columns(2)
                if c1.button(f"Победитель: {pa['full_name']}", key=f"win_a_{m['id']}"):
                    db.table("final_matches").update({"winner_id": pa["id"]}).eq(
                        "id", m["id"]
                    ).execute()
                    st.rerun()
                if c2.button(f"Победитель: {pb['full_name']}", key=f"win_b_{m['id']}"):
                    db.table("final_matches").update({"winner_id": pb["id"]}).eq(
                        "id", m["id"]
                    ).execute()
                    st.rerun()

elif mode == "Экспорт официального протокола (Excel)":
    st.title("Выгрузка сводных протоколов")
    teams = {t["id"]: t for t in (db.table("teams").select("*").execute().data or [])}
    parts = db.table("participants").select("*").execute().data or []

    rows = []
    for p in parts:
        t = teams.get(p["team_id"], {})
        rows.append(
            {
                "Муниципалитет": t.get("municipality", ""),
                "Школа": t.get("school_name", ""),
                "ФИО участника": p["full_name"],
                "Класс": p["class_name"],
                "Дата рождения": p["birth_date"],
                "Статус видео": p["video_status"],
                "Засчитано повторений": p.get("verified_reps") or 0,
                "Примечание судьи": p.get("judge_note") or "",
                "Ссылка Google Drive": p.get("video_url") or "",
            }
        )
    df_export = pd.DataFrame(rows)
    st.dataframe(df_export, use_container_width=True)
    csv_bytes = df_export.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 Скачать сводный протокол (CSV / Excel)",
        data=csv_bytes,
        file_name="protocol_pullups_vsosh.csv",
        mime="text/csv",
    )
