import io
import os
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive"]

@st.cache_resource
def get_drive_service():
    creds_info = dict(st.secrets["gdrive_service_account"])
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)

def get_or_create_folder(service, folder_name: str, parent_id: str) -> str:
    query = (
        f"name = '{folder_name}' and "
        f"mimeType = 'application/vnd.google-apps.folder' and "
        f"'{parent_id}' in parents and trashed = false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(body=metadata, fields="id").execute()
    return folder["id"]

def upload_participant_video(file_bytes: bytes, filename: str, mime_type: str, team_id: int, participant_id: int) -> dict:
    service = get_drive_service()
    root_folder_id = st.secrets["gdrive"]["root_folder_id"]
    teams_folder = get_or_create_folder(service, "teams", root_folder_id)
    team_folder = get_or_create_folder(service, str(team_id), teams_folder)
    part_folder = get_or_create_folder(service, str(participant_id), team_folder)

    file_metadata = {
        "name": filename,
        "parents": [part_folder],
    }
    media = MediaIoBaseUpload(io.BytesIO(file_bytes), mimetype=mime_type, resumable=True)
    uploaded = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, name, size, webViewLink"
    ).execute()
    return uploaded

def delete_drive_file(file_id: str):
    if not file_id:
        return
    try:
        service = get_drive_service()
        service.files().delete(fileId=file_id).execute()
    except Exception:
        pass

def download_video_to_local(file_id: str, dest_path: str) -> str:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return dest_path
