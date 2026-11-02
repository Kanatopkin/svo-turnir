import streamlit as st
import pandas as pd
from db import get_db, calculate_team_score
from gdrive import upload_participant_video, delete_drive_file

st.set_page_config(
    page_title="Турнир по подтягиваниям — Онлайн-портал",
    page_icon="🏅",
    layout="wide",
)

MAX_VIDEO_MB = 30
MAX_VIDEO_BYTES = MAX_VIDEO_MB * 1024 * 1024

st.sidebar.title("🏅 Турнир по подтягиваниям")
page = st.sidebar.radio(
    "Навигация",
    [
        "Главная и правила",
        "Кабинет команды (Регистрация)",
        "Турнирная таблица",
        "Финальная сетка (Топ-8)",
    ],
)

db = get_db()

if page == "Главная и правила":
    st.title("Школьные соревнования по подтягиваниям (8–9 классы)")
    st.markdown(
        """
        ### Формат проведения
        1. **Онлайн-регистрация и отборочный этап:** представитель школы формирует команду (от **10 до 12 участников** из 8–9 классов) и загружает видео каждого участника.
        2. **Хранение видеоматериалов:** видеоролики автоматически сохраняются в облачное хранилище **Google Drive** в структурную папку школы и спортсмена.
        3. **Судейство (ГСК):** Главная судейская коллегия проводит проверку видеозаписей в локальном судейском комплексе (с поддержкой компьютерного зрения YOLO-Pose и покадрового разбора).
        4. **Командный зачет:** сумма **10 лучших** подтвержденных результатов участников команды.
        5. **Личный финал:** 8 лучших участников по итогам отбора выходят в олимпийскую сетку плей-офф (четвертьфинал, полуфинал, матч за 3-е место и финал).

        ### Технические требования к видеозаписи
        * Формат файла: **MP4** или **MOV**, размер до **30 МБ**, разрешение не ниже **720p**.
        * В кадре должны четко читаться перекладина, фаза полного разгибания локтей в нижнем висе и пересечение грифа подбородком в верхней точке.
        """
    )

elif page == "Кабинет команды (Регистрация)":
    st.title("Кабинет представителя школы")

    if "team_session" not in st.session_state:
        st.session_state.team_session = None

    if st.session_state.team_session is None:
        tab_login, tab_register = st.tabs(["Вход в заявку школы", "Регистрация новой команды"])

        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email представителя / тренера")
                access_code = st.text_input("Пароль / код доступа команды", type="password")
                submitted = st.form_submit_button("Войти")
                if submitted:
                    res = (
                        db.table("teams")
                        .select("*")
                        .eq("coach_email", email.strip().lower())
                        .eq("access_code", access_code.strip())
                        .execute()
                    )
                    if res.data:
                        st.session_state.team_session = res.data[0]
                        st.rerun()
                    else:
                        st.error("Команда с таким Email и кодом доступа не найдена.")

        with tab_register:
            with st.form("register_team_form"):
                school_name = st.text_input("Наименование образовательной организации (школы)")
                municipality = st.text_input("Муниципалитет / Городской округ", value="г.о. Ступино")
                coach_name = st.text_input("ФИО представителя / учителя физической культуры")
                coach_phone = st.text_input("Контактный телефон")
                coach_email = st.text_input("Email для входа в кабинет")
                new_code = st.text_input("Придумайте пароль для входа в заявку", type="password")
                reg_btn = st.form_submit_button("Создать команду")

                if reg_btn:
                    if len(school_name.strip()) < 3 or len(coach_name.strip()) < 3 or len(new_code.strip()) < 4:
                        st.error("Заполните все обязательные поля (пароль не менее 4 символов).")
                    else:
                        existing = db.table("teams").select("id").eq("coach_email", coach_email.strip().lower()).execute()
                        if existing.data:
                            st.error("На этот Email уже зарегистрирована команда.")
                        else:
                            ins = (
                                db.table("teams")
                                .insert(
                                    {
                                        "school_name": school_name.strip(),
                                        "municipality": municipality.strip(),
                                        "coach_name": coach_name.strip(),
                                        "coach_phone": coach_phone.strip(),
                                        "coach_email": coach_email.strip().lower(),
                                        "access_code": new_code.strip(),
                                        "status": "draft",
                                        "is_published": False,
                                    }
                                )
                                .execute()
                            )
                            st.session_state.team_session = ins.data[0]
                            st.success("Команда успешно зарегистрирована!")
                            st.rerun()
    else:
        team_id = st.session_state.team_session["id"]
        team_res = db.table("teams").select("*").eq("id", team_id).execute()
        team = team_res.data[0]
        is_locked = team["status"] == "submitted"

        col_hdr, col_out = st.columns([4, 1])
        with col_hdr:
            st.subheader(f"{team['school_name']} ({team['municipality']})")
            st.caption(
                f"Представитель: {team['coach_name']} | Статус заявки: "
                f"**{'ОТПРАВЛЕНА В ГСК' if is_locked else 'ЧЕРНОВИК (Редактирование открыто)'}**"
            )
        with col_out:
            if st.button("Выйти из кабинета"):
                st.session_state.team_session = None
                st.rerun()

        roster_res = db.table("participants").select("*").eq("team_id", team_id).order("id").execute()
        roster = roster_res.data or []
        score, scoring_ids = calculate_team_score(roster)

        m1, m2, m3 = st.columns(3)
        m1.metric("Участников в заявке", f"{len(roster)} / 12 (мин. 10)")
        uploaded_count = sum(1 for p in roster if p.get("video_drive_id"))
        m2.metric("Загружено видео (Google Drive)", f"{uploaded_count} / {len(roster)}")
        m3.metric("Подтвержденный счет (Топ-10)", score)

        if not is_locked and len(roster) < 12:
            with st.expander("➕ Добавить участника в состав команды", expanded=len(roster) == 0):
                with st.form("add_participant_form", clear_on_submit=True):
                    c1, c2, c3 = st.columns([2, 1, 1])
                    full_name = c1.text_input("ФИО участника полностью")
                    class_name = c2.selectbox("Класс", ["8", "9"])
                    birth_date = c3.date_input("Дата рождения")

                    d1, d2, d3 = st.columns(3)
                    med = d1.checkbox("Меддопуск подтвержден")
                    par = d2.checkbox("Согласие родителей получено")
                    stu = d3.checkbox("Статус учащегося школы подтвержден")

                    add_p_btn = st.form_submit_button("Добавить в заявку")
                    if add_p_btn:
                        if len(full_name.strip()) < 5:
                            st.error("Укажите полное ФИО участника.")
                        elif not (med and par and stu):
                            st.error("Подтвердите наличие медицинского допуска, согласия родителей и статуса учащегося.")
                        else:
                            db.table("participants").insert(
                                {
                                    "team_id": team_id,
                                    "full_name": full_name.strip(),
                                    "class_name": class_name,
                                    "birth_date": str(birth_date),
                                    "medical_allowed": med,
                                    "parental_consent": par,
                                    "student_confirmed": stu,
                                    "video_status": "missing",
                                }
                            ).execute()
                            st.success(f"Участник {full_name} добавлен.")
                            st.rerun()

        st.markdown("### Состав команды и загрузка видео на Google Drive")
        for p in roster:
            with st.container(border=True):
                c_info, c_vid, c_act = st.columns([2, 2, 1])
                with c_info:
                    badge = "🟢 Зачет Топ-10" if p["id"] in scoring_ids else ""
                    st.markdown(f"**{p['full_name']}** ({p['class_name']} класс, д.р. {p['birth_date']}) {badge}")
                    status_map = {
                        "missing": "⚪ Видео не загружено",
                        "uploaded": "🔵 Видео на проверке ГСК",
                        "approved": f"✅ Зачтено: {p.get('verified_reps') or 0} повт.",
                        "rejected": "❌ Отклонено ГСК",
                    }
                    st.caption(f"Статус: {status_map.get(p['video_status'], p['video_status'])}")
                    if p.get("judge_note"):
                        st.info(f"Комментарий судьи: {p['judge_note']}")

                with c_vid:
                    if not is_locked:
                        v_file = st.file_uploader(
                            f"Видео MP4/MOV до {MAX_VIDEO_MB} МБ",
                            type=["mp4", "mov"],
                            key=f"up_{p['id']}",
                        )
                        req_ok = st.checkbox(
                            "Подтверждаю разрешение ≥720p и видимость перекладины",
                            key=f"chk_{p['id']}",
                        )
                        if v_file is not None and st.button("Загрузить в Google Drive", key=f"btn_up_{p['id']}"):
                            if not req_ok:
                                st.error("Подтвердите соблюдение технических требований к видео.")
                            elif v_file.size > MAX_VIDEO_BYTES:
                                st.error(f"Размер файла превышает {MAX_VIDEO_MB} МБ.")
                            else:
                                with st.spinner("Загрузка видео в Google Drive..."):
                                    safe_name = f"team{team_id}_p{p['id']}_{v_file.name}"
                                    uploaded = upload_participant_video(
                                        file_bytes=v_file.getvalue(),
                                        filename=safe_name,
                                        mime_type=v_file.type or "video/mp4",
                                        team_id=team_id,
                                        participant_id=p["id"],
                                    )
                                    if p.get("video_drive_id"):
                                        delete_drive_file(p["video_drive_id"])
                                    db.table("participants").update(
                                        {
                                            "video_drive_id": uploaded["id"],
                                            "video_name": uploaded["name"],
                                            "video_size": int(uploaded.get("size", v_file.size)),
                                            "video_url": uploaded.get("webViewLink", ""),
                                            "video_status": "uploaded",
                                            "verified_reps": None,
                                            "judge_note": None,
                                        }
                                    ).eq("id", p["id"]).execute()
                                    st.success("Видео сохранено на Google Drive!")
                                    st.rerun()
                    else:
                        if p.get("video_url"):
                            st.markdown(f"[🔗 Открыть видео на Google Drive]({p['video_url']})")

                with c_act:
                    if not is_locked:
                        if st.button("🗑 Удалить", key=f"del_{p['id']}"):
                            if p.get("video_drive_id"):
                                delete_drive_file(p["video_drive_id"])
                            db.table("participants").delete().eq("id", p["id"]).execute()
                            st.rerun()

        st.divider()
        if not is_locked:
            can_submit = (10 <= len(roster) <= 12) and all(bool(p.get("video_drive_id")) for p in roster)
            if st.button(
                "🚀 Отправить итоговую заявку в ГСК",
                type="primary",
                disabled=not can_submit,
            ):
                db.table("teams").update({"status": "submitted"}).eq("id", team_id).execute()
                st.success("Заявка официально отправлена в Главную судейскую коллегию!")
                st.rerun()
            if not can_submit:
                st.caption("Для отправки заявки необходимо от 10 до 12 участников с загруженным видео у каждого.")

elif page == "Турнирная таблица":
    st.title("Официальный командный зачет")
    teams_res = db.table("teams").select("*").eq("is_published", True).execute()
    pub_teams = teams_res.data or []

    if not pub_teams:
        st.info("Официальные протоколы команд еще не опубликованы судейской коллегией.")
    else:
        rows = []
        for t in pub_teams:
            p_res = db.table("participants").select("*").eq("team_id", t["id"]).execute()
            roster = p_res.data or []
            score, _ = calculate_team_score(roster)
            rows.append(
                {
                    "Школа": t["school_name"],
                    "Муниципалитет": t["municipality"],
                    "Участников в заявке": len(roster),
                    "Командный результат (Топ-10)": score,
                }
            )
        df = pd.DataFrame(rows).sort_values(by="Командный результат (Топ-10)", ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "Место"
        st.dataframe(df, use_container_width=True)

elif page == "Финальная сетка (Топ-8)":
    st.title("Личный финал (Олимпийская сетка Топ-8)")
    matches_res = db.table("final_matches").select("*").order("id").execute()
    matches = matches_res.data or []

    if not matches:
        st.info("Финальная сетка плей-офф формируется после завершения отборочного этапа.")
    else:
        p_all = {p["id"]: p for p in (db.table("participants").select("*").execute().data or [])}
        round_titles = {
            "quarterfinal": "1/4 финала (Четвертьфиналы)",
            "semifinal": "1/2 финала (Полуфиналы)",
            "third": "Матч за 3-е место",
            "final": "Финал за 1-е место",
        }
        for r_key, r_title in round_titles.items():
            r_matches = [m for m in matches if m["round"] == r_key]
            if r_matches:
                st.subheader(r_title)
                cols = st.columns(len(r_matches))
                for idx, m in enumerate(r_matches):
                    pa = p_all.get(m.get("participant_a_id"), {}).get("full_name", "Ожидание")
                    pb = p_all.get(m.get("participant_b_id"), {}).get("full_name", "Ожидание")
                    win = p_all.get(m.get("winner_id"), {}).get("full_name", "—")
                    with cols[idx]:
                        st.markdown(
                            f"**Матч #{m['match_no']}**  \n"
                            f"🅰️ {pa}  \n"
                            f"🅱️ {pb}  \n"
                            f"🏆 Победитель: **{win}**"
                        )
