import streamlit as st
import anthropic
import openai
import json
import re
import tempfile
import os
import uuid
import requests
from audio_recorder_streamlit import audio_recorder

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="English Coach 🎙️",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Wake Lock + CSS ───────────────────────────────────────────────────────────
st.markdown("""
<script>
(async () => {
  if ('wakeLock' in navigator) {
    let wakeLock = null;
    const request = async () => {
      try { wakeLock = await navigator.wakeLock.request('screen'); } catch {}
    };
    await request();
    document.addEventListener('visibilitychange', async () => {
      if (document.visibilityState === 'visible') await request();
    });
  }
})();
</script>
<style>
  .main { max-width: 720px; }
  .transcript-pill {
    background: #E6F1FB; border: 1px solid #B5D4F4;
    border-radius: 10px; padding: 8px 14px;
    font-size: 14px; color: #042C53; margin: 6px 0; display: block;
  }
  .correction-text {
    background: #FAEEDA; border: 1px solid #FAC775;
    border-radius: 10px; padding: 10px 14px;
    font-size: 13px; color: #633806; margin: 6px 0;
  }
  .correction-label {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: #633806; margin-bottom: 6px;
  }
  .wrong { color: #993C1D; text-decoration: line-through; }
  .right { color: #3B6D11; font-weight: 600; }
  .tip   { color: #633806; font-size: 12px; }
  .perfect {
    background: #EAF3DE; border: 1px solid #C0DD97;
    border-radius: 8px; padding: 8px 14px;
    color: #3B6D11; font-weight: 500; font-size: 14px; margin: 6px 0;
  }
</style>
""", unsafe_allow_html=True)

# ─── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a friendly English conversation partner for a French speaker at intermediate level.

After each user message, respond in this EXACT format — no deviation:

REPLY: [Your conversational response in English. 2-4 sentences. Warm, curious, ask a follow-up question.]
CORRECTIONS: [JSON array of mistakes. Empty array [] if English was perfect.]

JSON format for each correction:
{"wrong": "exact incorrect phrase", "right": "correct phrase", "tip": "brief explanation in French — calque du français, faux ami, faute de conjugaison, mauvaise préposition, article manquant, etc."}

Correct: grammar, wrong tense/conjugation, calques, false friends, missing/wrong articles, wrong prepositions, unnatural phrasing.
Do NOT correct: minor punctuation, capitalization.
Tone: encouraging. If perfect English → say so in REPLY and return [].
"""

# ─── API clients ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_anthropic():
    return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

@st.cache_resource
def get_openai():
    return openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ─── Supabase REST helpers ─────────────────────────────────────────────────────
def sb_headers(write=False):
    anon_key     = st.secrets["SUPABASE_KEY"]
    service_key  = st.secrets["SUPABASE_SERVICE_KEY"]
    auth_key     = service_key if write else anon_key
    return {
        "apikey":        anon_key,
        "Authorization": f"Bearer {auth_key}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }

def sb_url(path: str) -> str:
    base = st.secrets["SUPABASE_URL"].rstrip("/")
    return f"{base}/rest/v1/{path}"

def db_create_conversation(title: str) -> str:
    conv_id = str(uuid.uuid4())
    h = {**sb_headers(write=True), "Prefer": "return=minimal"}
    res = requests.post(sb_url("conversations"), headers=h, json={"id": conv_id, "title": title})
    if not res.ok:
        raise Exception(f"HTTP {res.status_code} — {res.text}")
    return conv_id

def db_update_conversation(conv_id: str):
    from datetime import datetime, timezone
    h = {**sb_headers(write=True), "Prefer": "return=minimal"}
    requests.patch(
        sb_url(f"conversations?id=eq.{conv_id}"),
        headers=h,
        json={"updated_at": datetime.now(timezone.utc).isoformat()},
    )

def db_save_message(conv_id: str, role: str, text: str, reply: str = None, corrections: list = None):
    h = {**sb_headers(write=True), "Prefer": "return=minimal"}
    res = requests.post(
        sb_url("messages"), headers=h,
        json={
            "conversation_id": conv_id,
            "role":            role,
            "text":            text,
            "reply":           reply,
            "corrections":     json.dumps(corrections) if corrections is not None else None,
        },
    )
    if not res.ok:
        raise Exception(f"HTTP {res.status_code} — {res.text}")
    db_update_conversation(conv_id)

def db_load_conversations() -> list:
    res = requests.get(
        sb_url("conversations?order=updated_at.desc"),
        headers=sb_headers(),
    )
    return res.json() if res.ok else []

def db_load_messages(conv_id: str) -> list:
    res = requests.get(
        sb_url(f"messages?conversation_id=eq.{conv_id}&order=created_at.asc"),
        headers=sb_headers(),
    )
    return res.json() if res.ok else []

def db_delete_conversation(conv_id: str):
    requests.delete(
        sb_url(f"conversations?id=eq.{conv_id}"),
        headers=sb_headers(),
    )

# ─── AI helpers ────────────────────────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes) -> str:
    client = get_openai()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1", file=f, language="en",
                prompt="The speaker is learning English and may make mistakes.",
            )
        return result.text.strip()
    finally:
        os.unlink(tmp_path)

def text_to_speech(text: str) -> bytes:
    response = get_openai().audio.speech.create(model="tts-1", voice="alloy", input=text)
    return response.content

def get_coach_response(user_text: str, history: list) -> str:
    messages = history + [{"role": "user", "content": user_text}]
    response = get_anthropic().messages.create(
        model="claude-opus-4-5", max_tokens=1000,
        system=SYSTEM_PROMPT, messages=messages,
    )
    return response.content[0].text

def parse_response(full_text: str) -> tuple[str, list]:
    reply_match = re.search(r"REPLY:\s*(.*?)(?=\nCORRECTIONS:|$)", full_text, re.DOTALL)
    corr_match  = re.search(r"CORRECTIONS:\s*([\s\S]*)", full_text)
    reply = reply_match.group(1).strip() if reply_match else full_text
    corrections = []
    if corr_match:
        try:
            corrections = json.loads(corr_match.group(1).strip())
        except Exception:
            corrections = []
    return reply, corrections

def corrections_to_speech(corrections: list) -> str | None:
    if not corrections:
        return None
    lines = [
        f"You said: {c.get('wrong','')}. The correct way is: {c.get('right','')}. {c.get('tip','')}"
        for c in corrections
    ]
    return " ... ".join(lines)

def render_corrections_text(corrections):
    if corrections is None:
        return
    if len(corrections) == 0:
        st.markdown('<div class="perfect">✓ No mistakes — perfect English! Keep it up 🎉</div>', unsafe_allow_html=True)
        return
    items = ""
    for i, c in enumerate(corrections):
        sep = "border-top:1px solid #FAC775;padding-top:8px;margin-top:8px;" if i > 0 else ""
        items += f"""<div style="{sep}">
            <div class="wrong">✗ {c.get('wrong','')}</div>
            <div class="right">✓ {c.get('right','')}</div>
            <div class="tip">💡 {c.get('tip','')}</div>
        </div>"""
    st.markdown(f"""<div class="correction-text">
        <div class="correction-label">✏️ Corrections</div>{items}
    </div>""", unsafe_allow_html=True)

def format_date(iso_str: str) -> str:
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%d/%m/%Y à %H:%M")
    except Exception:
        return iso_str

# ─── Session state ─────────────────────────────────────────────────────────────
if "view" not in st.session_state:
    st.session_state.view = "list"
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None
if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

# ══════════════════════════════════════════════════════════════════════════════
# VIEW : LIST
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.view == "list":
    st.title("🎙️ English Coach")
    st.caption("Tes conversations")

    if st.button("➕ Nouvelle conversation", use_container_width=True, type="primary"):
        st.session_state.update({
            "view": "chat", "conversation_id": None,
            "messages": [], "history": [],
            "last_audio_hash": None, "audio_key": 0,
        })
        st.rerun()

    st.divider()
    conversations = db_load_conversations()

    if not conversations:
        st.info("Aucune conversation pour l'instant. Lance-toi !")
    else:
        for conv in conversations:
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(
                    f"💬 {conv['title']}\n{format_date(conv['updated_at'])}",
                    key=f"open_{conv['id']}", use_container_width=True,
                ):
                    rows = db_load_messages(conv["id"])
                    messages, history = [], []
                    for row in rows:
                        if row["role"] == "user":
                            messages.append({"role": "user", "text": row["text"], "user_audio": None})
                            history.append({"role": "user", "content": row["text"]})
                        else:
                            corr = json.loads(row["corrections"]) if row["corrections"] else []
                            messages.append({
                                "role": "assistant", "reply": row["reply"] or "",
                                "corrections": corr, "corr_audio": None, "reply_audio": None,
                            })
                            history.append({"role": "assistant", "content": row["reply"] or ""})
                    st.session_state.update({
                        "view": "chat", "conversation_id": conv["id"],
                        "messages": messages, "history": history,
                        "last_audio_hash": None, "audio_key": 0,
                    })
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{conv['id']}", help="Supprimer"):
                    db_delete_conversation(conv["id"])
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# VIEW : CHAT
# ══════════════════════════════════════════════════════════════════════════════
else:
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("← Retour"):
            st.session_state.view = "list"
            st.rerun()
    with col2:
        title = st.session_state.messages[0]["text"][:40] + "…" if st.session_state.messages else "Nouvelle conversation"
        st.markdown(f"**{title}**")
        st.caption("English Coach · Corrections en temps réel")

    st.divider()

    if st.session_state.get("db_error"):
        st.error(f"⚠️ Erreur sauvegarde : {st.session_state.pop('db_error')}")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                if msg.get("user_audio"):
                    st.caption("🎤 Ta note vocale")
                    st.audio(msg["user_audio"], format="audio/wav")
                st.markdown(f'<div class="transcript-pill">📝 {msg["text"]}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                if msg.get("corrections") is not None:
                    if len(msg["corrections"]) > 0:
                        st.caption("🟠 Corrections")
                        if msg.get("corr_audio"):
                            st.audio(msg["corr_audio"], format="audio/mp3")
                        render_corrections_text(msg["corrections"])
                    else:
                        st.markdown('<div class="perfect">✓ No mistakes — perfect English! Keep it up 🎉</div>', unsafe_allow_html=True)
                st.caption("🔵 Réponse")
                if msg.get("reply_audio"):
                    st.audio(msg["reply_audio"], format="audio/mp3")
                st.markdown(msg["reply"])

    st.divider()
    st.markdown("**Appuie sur le micro, parle, rappuie pour envoyer**")

    audio_bytes = audio_recorder(
        text="", recording_color="#e74c3c", neutral_color="#378ADD",
        icon_name="microphone", icon_size="3x",
        pause_threshold=2.5, sample_rate=16_000,
        key=f"audio_{st.session_state.audio_key}",
    )

    if audio_bytes and len(audio_bytes) > 2000:
        audio_hash = hash(audio_bytes)
        if st.session_state.last_audio_hash != audio_hash:
            st.session_state.last_audio_hash = audio_hash

            with st.spinner("Transcription..."):
                try:
                    user_text = transcribe_audio(audio_bytes)
                except Exception as e:
                    st.error(f"Erreur de transcription : {e}")
                    user_text = None

            if user_text:
                st.info(f"🎤 Transcrit : **{user_text}**")
                with st.spinner("Analyse, corrections et synthèse vocale..."):
                    try:
                        raw = get_coach_response(user_text, st.session_state.history)
                    except Exception as e:
                        st.error(f"Erreur Claude : {e}")
                        raw = None

                    if raw:
                        try:
                            reply, corrections = parse_response(raw)
                            corr_audio  = text_to_speech(corrections_to_speech(corrections)) if corrections else None
                            reply_audio = text_to_speech(reply)
                        except Exception as e:
                            st.error(f"Erreur TTS : {e}")
                            reply, corrections, corr_audio, reply_audio = raw, [], None, None

                        try:
                            if st.session_state.conversation_id is None:
                                conv_id = db_create_conversation(user_text[:60])
                                st.session_state.conversation_id = conv_id
                            else:
                                conv_id = st.session_state.conversation_id
                            db_save_message(conv_id, "user", user_text)
                            db_save_message(conv_id, "assistant", reply, reply=reply, corrections=corrections)
                        except Exception as e:
                            st.session_state["db_error"] = str(e)

                        st.session_state.history += [
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": raw},
                        ]
                        st.session_state.messages += [
                            {"role": "user", "text": user_text, "user_audio": audio_bytes},
                            {"role": "assistant", "reply": reply, "corrections": corrections,
                             "corr_audio": corr_audio, "reply_audio": reply_audio},
                        ]
                        st.session_state.audio_key += 1
                        st.rerun()

    with st.expander("✏️ Ou tape en anglais (mode texte)"):
        col1, col2 = st.columns([5, 1])
        with col1:
            text_input = st.text_input("Type in English...", key="text_fallback", label_visibility="collapsed")
        with col2:
            send = st.button("Send", use_container_width=True)

        if send and text_input.strip():
            user_text = text_input.strip()
            with st.spinner("Analyse, corrections et synthèse vocale..."):
                try:
                    raw = get_coach_response(user_text, st.session_state.history)
                    reply, corrections = parse_response(raw)
                    corr_audio  = text_to_speech(corrections_to_speech(corrections)) if corrections else None
                    reply_audio = text_to_speech(reply)

                    if st.session_state.conversation_id is None:
                        conv_id = db_create_conversation(user_text[:60])
                        st.session_state.conversation_id = conv_id
                    else:
                        conv_id = st.session_state.conversation_id

                    db_save_message(conv_id, "user", user_text)
                    db_save_message(conv_id, "assistant", reply, reply=reply, corrections=corrections)

                    st.session_state.history += [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": raw},
                    ]
                    st.session_state.messages += [
                        {"role": "user", "text": user_text, "user_audio": None},
                        {"role": "assistant", "reply": reply, "corrections": corrections,
                         "corr_audio": corr_audio, "reply_audio": reply_audio},
                    ]
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur : {e}")
