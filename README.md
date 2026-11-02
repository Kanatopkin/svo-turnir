# Турнирная платформа по подтягиваниям (Streamlit + Google Drive + Локальное AI-судейство)

Проект перенесен со стека React/tRPC/MySQL на **Python + Streamlit** с разделением на два контура:
1. **Онлайн-портал (`app.py`)** — деплоится из GitHub в **Streamlit Community Cloud** для регистрации школьных команд (10–12 человек, 8–9 классы), загрузки видео напрямую на **Google Drive** и публикации онлайн-таблиц.
2. **Локальный пульт судьи (`local_judge.py`)** — запускается на компьютере Главной судейской коллегии, скачивает видео с Google Drive в локальный кэш и проверяет технику выполнения подтягиваний с помощью **Ultralytics YOLO-Pose** и OpenCV.

## Быстрый старт

### 1. База данных (Supabase PostgreSQL)
Выполните SQL-скрипт из файла `schema_supabase.sql` в SQL Editor вашего проекта Supabase.

### 2. Настройка Google Drive API
1. В Google Cloud Console создайте Service Account и скачайте JSON-ключ.
2. На своем Google Диске создайте корневую папку для видео соревнований и откройте к ней доступ (Редактор) для `client_email` сервисного аккаунта.
3. Скопируйте ID этой папки из адресной строки браузера в `root_folder_id`.

### 3. Конфигурация секретов
Скопируйте `.streamlit/secrets.toml.example` в `.streamlit/secrets.toml` и заполните ключи Supabase и Google Drive.

### 4. Запуск
* **Онлайн-портал (или деплой на share.streamlit.io через GitHub):**
  ```bash
  streamlit run app.py
  ```
* **Локальное рабочее место ГСК (с поддержкой YOLO-Pose):**
  ```bash
  pip install -r requirements.txt
  streamlit run local_judge.py
  ```
