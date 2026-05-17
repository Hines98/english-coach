import streamlit as st
import anthropic
import openai
import json
import re
import tempfile
import os
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
// Wake Lock API — keeps screen on while app is open (Chrome/Android)
(async () => {
  if ('wakeLock' in navigator) {
    let wakeLock = null;
    const requestWakeLock = async () => {
      try {
        wakeLock = await navigator.wakeLock.request('screen');
      } catch (err) {
        console.log('Wake Lock not available:', err);
      }
    };
    await requestWakeLock();
    document.addEventListener('visibilitychange', async () => {
      if (document.visibilityState === 'visible') {
        await requestWakeLock();
      }
    });
  }
})();
</script>

<style>
    .main { max-width: 720px; }
    .transcript-pill {
        background: #E6F1FB;
        border: 1px solid #B5D4F4;
        border-radius: 10px;
        padding: 8px 14px;
        font-size: 14px;
        color: #042C53;
        margin: 6px 0;
        display: block;
    }
    .correction-text {
        background: #FAEEDA;
        border: 1px solid #FAC775;
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 13px;
        color: #633806;
        margin: 6px 0;
    }
    .correction-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #633806;
        margin-bottom: 6px;
    }
    .wrong { color: #993C1D; text-decoration: line-through; }
    .right { color: #3B6D11; font-weight: 600; }
    .tip   { color: #633806; font-size: 12px; }
    .perfect {
        background: #EAF3DE;
        border: 1px solid #C0DD97;
        border-radius: 8px;
        padding: 8px 14px;
        color: #3B6D11;
        font-weight: 500;
        font-size: 14px;
        margin: 6px 0;
    }
    div[data-testid="stChatMessage"] { padding: 6px 0; }
</style>
""", unsafe_allow_html=True)

# ─── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a friendly English conversation partner for a French speaker at intermediate level.

After each user message, respond in this EXACT format — no deviation:

REPLY: [Your conversational response in English. 2-4 sentences. Warm, curious, ask a follow-up question.]
CORRECTIONS: [JSON array of mistakes. Empty array [] if English was perfect.]

JSON format for each correction:
{"wrong": "exact incorrect phrase", "right": "correct phrase", "tip": "brief explanation in French — calque du français, faux ami, faute de conjugaison, mauvaise préposition, article manquant, etc."}

Correct: grammar, wrong tense/conjugation, literal French-to-English translations (calques), false friends, missing/wrong articles, wrong prepositions, unnatural phrasing.
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

# ─── Core functions ────────────────────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes) -> str:
    client = get_openai()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en",
                prompt="The speaker is learning English and may make mistakes.",
            )
        return result.text.strip()
    finally:
        os.unlink(tmp_path)


def text_to_speech(text: str) -> bytes:
    client = get_openai()
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    return response.content


def get_coach_response(user_text: str, history: list) -> str:
    client = get_anthropic()
    messages = history + [{"role": "user", "content": user_text}]
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=messages,
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


def corrections_to_speech_text(corrections: list) -> str | None:
    """Convert corrections list to a spoken sentence for TTS."""
    if not corrections:
        return None
    lines = []
    for c in corrections:
        lines.append(
            f"You said: {c.get('wrong', '')}. "
            f"The correct way is: {c.get('right', '')}. "
            f"{c.get('tip', '')}"
        )
    return " ... ".join(lines)


def render_corrections_text(corrections: list | None):
    """Render corrections as readable text (shown alongside the audio)."""
    if corrections is None:
        return
    if len(corrections) == 0:
        st.markdown('<div class="perfect">✓ No mistakes — perfect English! Keep it up 🎉</div>', unsafe_allow_html=True)
        return
    items_html = ""
    for i, c in enumerate(corrections):
        sep = "border-top: 1px solid #FAC775; padding-top: 8px; margin-top: 8px;" if i > 0 else ""
        items_html += f"""<div style="{sep}">
            <div class="wrong">✗ {c.get('wrong','')}</div>
            <div class="right">✓ {c.get('right','')}</div>
            <div class="tip">💡 {c.get('tip','')}</div>
        </div>"""
    st.markdown(f"""
    <div class="correction-text">
        <div class="correction-label">✏️ Corrections</div>
        {items_html}
    </div>""", unsafe_allow_html=True)


# ─── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None
if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

# ─── Header ────────────────────────────────────────────────────────────────────
st.title("🎙️ English Coach")
st.caption("Parle en anglais, je corrige en temps réel · Niveau intermédiaire")

if st.button("🗑️ Nouvelle conversation", use_container_width=False):
    st.session_state.messages = []
    st.session_state.history  = []
    st.session_state.last_audio_hash = None
    st.session_state.audio_key = 0
    st.rerun()

st.divider()

# ─── Conversation display ──────────────────────────────────────────────────────
for msg in st.session_state.messages:

    if msg["role"] == "user":
        with st.chat_message("user"):
            # User's own recording — réécoute
            if msg.get("user_audio"):
                st.caption("🎤 Ta note vocale")
                st.audio(msg["user_audio"], format="audio/wav")
            # Transcription
            st.markdown(
                f'<div class="transcript-pill">📝 {msg["text"]}</div>',
                unsafe_allow_html=True
            )

    else:
        with st.chat_message("assistant"):
            # ── Corrections ──────────────────────────────────────────
            if msg.get("corrections") is not None:
                if len(msg["corrections"]) > 0:
                    st.caption("🟠 Corrections")
                    st.audio(msg["corr_audio"], format="audio/mp3")
                    render_corrections_text(msg["corrections"])
                else:
                    st.markdown(
                        '<div class="perfect">✓ No mistakes — perfect English! Keep it up 🎉</div>',
                        unsafe_allow_html=True
                    )

            # ── Réponse conversationnelle ─────────────────────────────
            st.caption("🔵 Réponse")
            st.audio(msg["reply_audio"], format="audio/mp3")
            st.markdown(msg["reply"])

# ─── Input section ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("**Appuie sur le micro, parle, rappuie pour envoyer**")

audio_bytes = audio_recorder(
    text="",
    recording_color="#e74c3c",
    neutral_color="#378ADD",
    icon_name="microphone",
    icon_size="3x",
    pause_threshold=2.5,
    sample_rate=16_000,
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
                    reply, corrections = parse_response(raw)

                    # TTS corrections (only if there are mistakes)
                    corr_audio = None
                    corr_speech = corrections_to_speech_text(corrections)
                    if corr_speech:
                        corr_audio = text_to_speech(corr_speech)

                    # TTS reply
                    reply_audio = text_to_speech(reply)

                    st.session_state.history += [
                        {"role": "user",      "content": user_text},
                        {"role": "assistant", "content": raw},
                    ]
                    st.session_state.messages += [
                        {
                            "role":       "user",
                            "text":       user_text,
                            "user_audio": audio_bytes,
                        },
                        {
                            "role":        "assistant",
                            "reply":       reply,
                            "corrections": corrections,
                            "corr_audio":  corr_audio,
                            "reply_audio": reply_audio,
                        },
                    ]
                    st.session_state.audio_key += 1
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur : {e}")

# ─── Text fallback ─────────────────────────────────────────────────────────────
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

                corr_audio = None
                corr_speech = corrections_to_speech_text(corrections)
                if corr_speech:
                    corr_audio = text_to_speech(corr_speech)

                reply_audio = text_to_speech(reply)

                st.session_state.history += [
                    {"role": "user",      "content": user_text},
                    {"role": "assistant", "content": raw},
                ]
                st.session_state.messages += [
                    {
                        "role":       "user",
                        "text":       user_text,
                        "user_audio": None,
                    },
                    {
                        "role":        "assistant",
                        "reply":       reply,
                        "corrections": corrections,
                        "corr_audio":  corr_audio,
                        "reply_audio": reply_audio,
                    },
                ]
                st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")
