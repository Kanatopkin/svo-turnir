import streamlit as st
from supabase import create_client, Client

@st.cache_resource
def get_db() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

def calculate_team_score(participants: list) -> tuple[int, list[int]]:
    """Сумма 10 лучших подтвержденных результатов в команде (из максимум 12 участников)."""
    approved = [
        p for p in participants
        if p.get("video_status") == "approved" and p.get("verified_reps") is not None
    ]
    sorted_p = sorted(approved, key=lambda x: x.get("verified_reps") or 0, reverse=True)
    top_10 = sorted_p[:10]
    score = sum(p.get("verified_reps") or 0 for p in top_10)
    scoring_ids = [p["id"] for p in top_10]
    return score, scoring_ids
