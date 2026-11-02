import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AuthenticationError, OpenAI, OpenAIError

# =============================================================================
# Styles & animations — DocuNest AI (light, CSS-only, no images)
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');

:root {
    --bg: #F7F7F1;
    --bg-2: #EEF1E8;
    --card: rgba(255, 255, 255, 0.72);
    --primary: #71866B;
    --primary-soft: #A5B89A;
    --olive: #8B9A7A;
    --text: #28332B;
    --text-2: #737D73;
    --border: #E1E7DC;
    --shadow: 0 18px 50px rgba(40, 51, 43, 0.10);
    --shadow-hover: 0 22px 56px rgba(113, 134, 107, 0.18);
}

/* Outfit for UI text — NEVER on Material icon buttons (causes "visibility" text) */
html, body, .stApp, .stMarkdown, .stMarkdown p, label, li,
h1, h2, h3, h4, h5, h6, input, textarea,
[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"],
.stButton > button,
.stFormSubmitButton > button,
.stLinkButton > a {
    font-family: 'Outfit', sans-serif !important;
}

.stApp { color: var(--text); }
.stApp p, .stApp label, .stApp li, .stApp small,
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stCaptionContainer"] p {
    color: var(--text) !important;
    opacity: 1 !important;
}
.security-note, .section-sub, .brand-tagline, .login-foot {
    color: var(--text-2) !important;
}

/* ==========================================================================
   Full-viewport animated mesh background
   ========================================================================== */
.stApp {
    background:
        radial-gradient(ellipse 70% 55% at 8% 12%, rgba(165, 184, 154, 0.55), transparent 60%),
        radial-gradient(ellipse 55% 45% at 92% 8%, rgba(232, 237, 227, 0.95), transparent 55%),
        radial-gradient(ellipse 50% 50% at 78% 88%, rgba(113, 134, 107, 0.28), transparent 60%),
        radial-gradient(ellipse 40% 35% at 20% 80%, rgba(139, 154, 122, 0.22), transparent 55%),
        linear-gradient(125deg, #F7F7F1 0%, #EEF1E8 35%, #E8EDE3 65%, #F4F5EE 100%);
    background-size: 140% 140%, 120% 120%, 130% 130%, 110% 110%, 240% 240%;
    animation: meshDrift 28s ease-in-out infinite;
    min-height: 100vh;
}
@keyframes meshDrift {
    0%, 100% { background-position: 0% 0%, 100% 0%, 80% 100%, 10% 90%, 0% 40%; }
    33%      { background-position: 40% 20%, 60% 30%, 40% 60%, 30% 70%, 50% 60%; }
    66%      { background-position: 20% 40%, 30% 10%, 70% 40%, 50% 40%, 100% 50%; }
}

/* Extra orbs via fixed layer */
.bg-scene {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
}
.bg-scene .orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(70px);
    opacity: 0.72;
    will-change: transform;
}
.bg-scene .orb-a {
    width: 560px; height: 560px;
    background: radial-gradient(circle, #A5B89A 0%, rgba(165,184,154,0.35) 45%, transparent 70%);
    top: -160px; left: -140px;
    animation: orbFloatA 22s ease-in-out infinite;
}
.bg-scene .orb-b {
    width: 500px; height: 500px;
    background: radial-gradient(circle, #C5D2BC 0%, rgba(214,224,207,0.5) 40%, transparent 70%);
    top: 12%; right: -120px;
    animation: orbFloatB 26s ease-in-out infinite;
}
.bg-scene .orb-c {
    width: 440px; height: 440px;
    background: radial-gradient(circle, #8B9A7A 0%, rgba(113,134,107,0.35) 40%, transparent 72%);
    bottom: -120px; left: 22%;
    opacity: 0.5;
    animation: orbFloatC 24s ease-in-out infinite;
}
.bg-scene .orb-d {
    width: 320px; height: 320px;
    background: radial-gradient(circle, #F7F7F1 0%, #C5D2BC 45%, transparent 75%);
    top: 52%; left: 6%;
    opacity: 0.6;
    animation: orbFloatA 19s ease-in-out infinite reverse;
}
.bg-scene .blob {
    position: absolute;
    width: 340px; height: 240px;
    background: linear-gradient(135deg, rgba(165,184,154,0.35), rgba(247,247,241,0.1));
    border-radius: 60% 40% 50% 50% / 50% 60% 40% 50%;
    filter: blur(40px);
    animation: blobMorph 20s ease-in-out infinite;
}
.bg-scene .blob-1 { top: 12%; left: 42%; }
.bg-scene .blob-2 {
    bottom: 18%; right: 18%;
    width: 300px; height: 220px;
    animation-duration: 26s;
    animation-direction: reverse;
    background: linear-gradient(160deg, rgba(113,134,107,0.22), rgba(238,241,232,0.15));
}
.bg-scene .glow {
    position: absolute;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.55) 0%, transparent 70%);
    filter: blur(20px);
    animation: glowPulse 10s ease-in-out infinite;
}
.bg-scene .glow-1 { top: 30%; left: 60%; }
.bg-scene .glow-2 { bottom: 28%; left: 35%; animation-delay: -4s; }

/* CSS particles (box dots, no images) */
.bg-scene .particle {
    position: absolute;
    width: 5px; height: 5px;
    border-radius: 50%;
    background: rgba(113, 134, 107, 0.45);
    box-shadow:
        40px 60px 0 rgba(165, 184, 154, 0.35),
        120px 20px 0 rgba(113, 134, 107, 0.25),
        200px 90px 0 rgba(139, 154, 122, 0.3),
        280px 40px 0 rgba(165, 184, 154, 0.22),
        360px 110px 0 rgba(113, 134, 107, 0.28),
        80px 140px 0 rgba(232, 237, 227, 0.7),
        160px 180px 0 rgba(113, 134, 107, 0.2),
        240px 150px 0 rgba(165, 184, 154, 0.35),
        320px 200px 0 rgba(139, 154, 122, 0.25),
        400px 70px 0 rgba(113, 134, 107, 0.18);
    animation: particleDrift 32s linear infinite;
}
.bg-scene .particle-a { top: 15%; left: 10%; animation-duration: 36s; }
.bg-scene .particle-b { top: 45%; left: 55%; animation-duration: 42s; animation-delay: -8s; opacity: 0.7; }
.bg-scene .particle-c { top: 70%; left: 25%; animation-duration: 38s; animation-delay: -14s; }

@keyframes orbFloatA {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(110px, 80px) scale(1.14); }
}
@keyframes orbFloatB {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(-90px, 70px) scale(1.1); }
}
@keyframes orbFloatC {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(70px, -90px) scale(1.18); }
}
@keyframes blobMorph {
    0%, 100% { transform: translate(0,0) rotate(0deg); border-radius: 60% 40% 50% 50% / 50% 60% 40% 50%; }
    50% { transform: translate(40px, -30px) rotate(12deg); border-radius: 40% 60% 45% 55% / 55% 40% 60% 45%; }
}
@keyframes glowPulse {
    0%, 100% { opacity: 0.35; transform: scale(1); }
    50% { opacity: 0.7; transform: scale(1.25); }
}
@keyframes particleDrift {
    0% { transform: translate(0, 0); }
    50% { transform: translate(50px, -40px); }
    100% { transform: translate(0, 0); }
}

.stApp::before, .stApp::after { content: none !important; }

header[data-testid="stHeader"],
[data-testid="stToolbar"] {
    background: transparent !important;
}

.block-container {
    position: relative;
    z-index: 1;
    padding-top: 1.1rem !important;
    padding-bottom: 1.5rem !important;
    max-width: 1100px;
    animation: fadeUp 0.65s ease both;
}
[data-testid="stForm"] {
    width: 100% !important;
    max-width: 520px;
    margin-left: auto !important;
    margin-right: auto !important;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideIn {
    from { opacity: 0; transform: translateX(-16px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes softPulse {
    0%, 100% { box-shadow: 0 8px 22px rgba(113, 134, 107, 0.28); }
    50% { box-shadow: 0 8px 22px rgba(113, 134, 107, 0.28), 0 0 0 8px rgba(113, 134, 107, 0.12); }
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes cardIn {
    from { opacity: 0; transform: translateY(28px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes statusBlink {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(113,134,107,0.45); }
    50% { opacity: 0.75; box-shadow: 0 0 0 5px rgba(113,134,107,0); }
}

/* ==========================================================================
   Branding — solid elegant type (no selection-like gradient fill)
   ========================================================================== */
.brand-wrap {
    text-align: center;
    margin: 0 auto 0.85rem;
    animation: fadeUp 0.7s ease both;
}
.brand-mark {
    font-size: 2.05rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 0;
    color: #28332B !important;
    -webkit-text-fill-color: #28332B !important;
    background: none !important;
    user-select: none;
}
.brand-mark .ai {
    color: #71866B !important;
    -webkit-text-fill-color: #71866B !important;
    font-weight: 600;
}
.brand-tagline {
    margin: 0.4rem 0 0;
    color: var(--text-2) !important;
    font-size: 1rem;
    font-weight: 500;
}

/* CSS-only document mark */
.doc-logo {
    width: 52px;
    height: 62px;
    margin: 0 auto 0.85rem;
    position: relative;
    border-radius: 10px 10px 8px 8px;
    background: linear-gradient(160deg, #FFFFFF 0%, #EEF1E8 100%);
    border: 1.5px solid #C5D2BC;
    box-shadow: 0 10px 28px rgba(113, 134, 107, 0.18);
    animation: fadeUp 0.6s ease both, softPulse 4s ease-in-out infinite;
}
.doc-logo::before {
    content: "";
    position: absolute;
    top: 0; right: 0;
    width: 16px; height: 16px;
    background: linear-gradient(135deg, transparent 50%, #A5B89A 50%);
    border-radius: 0 8px 0 0;
}
.doc-logo span {
    position: absolute;
    left: 10px; right: 10px;
    height: 3px;
    border-radius: 2px;
    background: #A5B89A;
}
.doc-logo span:nth-child(1) { top: 22px; }
.doc-logo span:nth-child(2) { top: 30px; width: 60%; right: auto; }
.doc-logo span:nth-child(3) { top: 38px; width: 75%; right: auto; }

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.65rem;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.55);
    border: 1px solid var(--border);
    backdrop-filter: blur(8px);
    color: var(--primary) !important;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.status-pill .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #71866B;
    animation: statusBlink 2.2s ease-in-out infinite;
}

/* ==========================================================================
   Glass login card
   ========================================================================== */
.login-shell {
    max-width: 440px;
    margin: 0 auto;
    padding: 1.65rem 1.7rem 1.35rem;
    background: rgba(255, 255, 255, 0.68);
    border: 1px solid rgba(225, 231, 220, 0.95);
    border-radius: 24px;
    box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.85);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    animation: cardIn 0.75s cubic-bezier(.22,.9,.28,1) both;
}
.login-shell .card-title {
    margin: 0 0 0.3rem;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.login-shell .card-desc {
    margin: 0 0 0.15rem;
    color: var(--text-2) !important;
    font-size: 0.92rem;
    line-height: 1.45;
}

.dn-card, .glass, .upload-shell, .doc-chip {
    background: rgba(255,255,255,0.78) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    box-shadow: var(--shadow);
    backdrop-filter: blur(14px);
}
/* Login form = glass card */
[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.70) !important;
    border: 1px solid rgba(225, 231, 220, 0.95) !important;
    border-radius: 24px !important;
    box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.85) !important;
    backdrop-filter: blur(18px) !important;
    -webkit-backdrop-filter: blur(18px) !important;
    padding: 1.55rem 1.55rem 1.25rem !important;
    animation: cardIn 0.75s cubic-bezier(.22,.9,.28,1) both;
}
.glass {
    padding: 1.25rem 1.4rem;
    text-align: center;
    color: var(--text) !important;
    font-weight: 500;
}
.security-note, .login-foot {
    text-align: center;
    margin-top: 0.85rem !important;
    font-size: 0.8rem !important;
    color: var(--text-2) !important;
    line-height: 1.4;
}
.login-link-row {
    text-align: center;
    margin: 0.85rem 0 0.35rem;
    color: var(--text-2) !important;
    font-size: 0.9rem;
}

/* ==========================================================================
   API key input + native eye toggle (fixed)
   ========================================================================== */
.stTextInput label, .stSelectbox label, .stSlider label,
.stFileUploader label, [data-testid="stWidgetLabel"] p {
    color: var(--text) !important;
    font-weight: 600 !important;
}

/* Keep help (?) icon usable — do not restyle every text-input button */
[data-testid="stTextInput"] > div,
[data-testid="stTextInput"] [data-baseweb="base-input"],
[data-testid="stTextInput"] [data-baseweb="input"],
div[data-baseweb="base-input"] {
    position: relative !important;
}

.stTextInput input,
.stTextInput input[type="password"],
.stTextInput input[type="text"] {
    background: rgba(255,255,255,0.92) !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    caret-color: var(--text) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 0.85rem 3.25rem 0.85rem 1rem !important;
    min-height: 3rem !important;
    font-size: 0.95rem !important;
    transition: border-color 0.22s ease, box-shadow 0.22s ease !important;
}
.stTextInput input::placeholder {
    color: #9AA39A !important;
    -webkit-text-fill-color: #9AA39A !important;
    opacity: 1 !important;
}
.stTextInput input:focus {
    border-color: #71866B !important;
    box-shadow: 0 0 0 3px rgba(113, 134, 107, 0.22) !important;
    outline: none !important;
}

/*
 * Password show/hide control:
 * Streamlit renders Material ligature text ("visibility") which often fails to
 * glyph correctly. Hide that text and draw a CSS-only eye icon instead.
 */
button[aria-label="Show password"],
button[aria-label="Hide password"] {
    position: absolute !important;
    right: 8px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    width: 2.4rem !important;
    height: 2.4rem !important;
    min-width: 2.4rem !important;
    min-height: 2.4rem !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    border-radius: 10px !important;
    background: transparent !important;
    background-image: none !important;
    box-shadow: none !important;
    color: transparent !important;
    -webkit-text-fill-color: transparent !important;
    font-size: 0 !important;
    line-height: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    z-index: 5 !important;
    animation: none !important;
    filter: none !important;
    cursor: pointer !important;
}
button[aria-label="Show password"] *,
button[aria-label="Hide password"] * {
    font-size: 0 !important;
    line-height: 0 !important;
    width: 0 !important;
    height: 0 !important;
    max-width: 0 !important;
    max-height: 0 !important;
    opacity: 0 !important;
    visibility: hidden !important;
    overflow: hidden !important;
    color: transparent !important;
    -webkit-text-fill-color: transparent !important;
    position: absolute !important;
    pointer-events: none !important;
}
/* CSS eye (open) */
button[aria-label="Show password"]::before {
    content: "" !important;
    display: block !important;
    width: 1.15rem !important;
    height: 0.72rem !important;
    border: 2px solid #71866B !important;
    border-radius: 50% !important;
    box-sizing: border-box !important;
    background: transparent !important;
    position: relative !important;
    flex-shrink: 0 !important;
}
button[aria-label="Show password"]::after {
    content: "" !important;
    position: absolute !important;
    width: 0.34rem !important;
    height: 0.34rem !important;
    background: #71866B !important;
    border-radius: 50% !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
}
/* CSS eye (hidden / struck) */
button[aria-label="Hide password"]::before {
    content: "" !important;
    display: block !important;
    width: 1.15rem !important;
    height: 0.72rem !important;
    border: 2px solid #71866B !important;
    border-radius: 50% !important;
    box-sizing: border-box !important;
    background: transparent !important;
    flex-shrink: 0 !important;
    opacity: 0.85 !important;
}
button[aria-label="Hide password"]::after {
    content: "" !important;
    position: absolute !important;
    width: 1.35rem !important;
    height: 2px !important;
    background: #71866B !important;
    border-radius: 2px !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) rotate(-40deg) !important;
}
button[aria-label="Show password"]:hover,
button[aria-label="Hide password"]:hover {
    background: rgba(113, 134, 107, 0.12) !important;
    background-image: none !important;
    transform: translateY(-50%) !important;
    box-shadow: none !important;
    filter: none !important;
}
button[aria-label="Show password"]:active,
button[aria-label="Hide password"]:active {
    transform: translateY(-50%) scale(0.96) !important;
}

[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 12px !important;
}
[data-baseweb="popover"] li, [data-baseweb="menu"] li {
    color: var(--text) !important;
}

/* ==========================================================================
   Buttons
   ========================================================================== */
.stButton > button,
.stFormSubmitButton > button,
.stLinkButton > a {
    border-radius: 14px !important;
    font-weight: 600 !important;
    border: none !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    background: linear-gradient(120deg, #5F7359 0%, #71866B 35%, #8FA585 70%, #A5B89A 100%) !important;
    background-size: 220% 220% !important;
    animation: gradientShift 6s ease infinite;
    box-shadow: 0 10px 26px rgba(113, 134, 107, 0.30) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease !important;
    min-height: 2.85rem !important;
}
.stFormSubmitButton > button {
    animation: gradientShift 5s ease infinite, softPulse 3s ease-in-out infinite;
    letter-spacing: 0.01em !important;
    font-size: 1.02rem !important;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stLinkButton > a:hover {
    transform: translateY(-3px);
    filter: brightness(1.04);
    box-shadow: 0 16px 34px rgba(113, 134, 107, 0.38) !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
.stButton > button:active,
.stFormSubmitButton > button:active {
    transform: translateY(0) scale(0.985);
}
.stButton > button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
    background: rgba(255,255,255,0.75) !important;
    background-image: none !important;
    animation: none !important;
    color: var(--primary) !important;
    -webkit-text-fill-color: var(--primary) !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 6px 18px rgba(40,51,43,0.06) !important;
}
.stButton > button[kind="secondary"]:hover {
    color: var(--primary) !important;
    -webkit-text-fill-color: var(--primary) !important;
    border-color: var(--primary-soft) !important;
    background: #FFFFFF !important;
}

/* Signup steps */
.step {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    background: rgba(255,255,255,0.78);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: var(--shadow);
    backdrop-filter: blur(10px);
    animation: slideIn 0.55s ease both;
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
}
.step:hover {
    transform: translateY(-3px) translateX(4px);
    box-shadow: var(--shadow-hover);
    border-color: var(--primary-soft);
}
.step:nth-child(1) { animation-delay: 0.05s; }
.step:nth-child(2) { animation-delay: 0.14s; }
.step:nth-child(3) { animation-delay: 0.23s; }
.step:nth-child(4) { animation-delay: 0.32s; }
.step .num {
    min-width: 36px;
    height: 36px;
    border-radius: 50%;
    background: linear-gradient(145deg, #71866B, #A5B89A);
    color: #FFFFFF !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.95rem;
    box-shadow: 0 6px 16px rgba(113, 134, 107, 0.28);
}
.step b { color: var(--text) !important; font-size: 0.98rem; }
.step span { color: var(--text-2) !important; font-size: 0.9rem; line-height: 1.4; }

/* Chat */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.82) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: var(--shadow);
    animation: fadeUp 0.4s ease both;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] li { color: var(--text) !important; }
[data-testid="stChatInput"] { border-radius: 16px !important; }
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] div[contenteditable="true"] {
    color: var(--text) !important;
    caret-color: var(--text) !important;
}
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.82) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] p { color: var(--text) !important; }

.typing { display: inline-flex; gap: 7px; padding: 8px 4px; align-items: center; }
.typing i {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--primary);
    animation: blink 1.2s infinite ease-in-out;
}
.typing i:nth-child(2) { animation-delay: 0.18s; background: var(--primary-soft); }
.typing i:nth-child(3) { animation-delay: 0.36s; }
@keyframes blink {
    0%, 80%, 100% { transform: scale(0.55); opacity: 0.35; }
    40% { transform: scale(1); opacity: 1; }
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(238,241,232,0.95) 0%, rgba(247,247,241,0.95) 100%) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: blur(12px);
}
[data-testid="stSidebar"] > div:first-child { animation: slideIn 0.55s ease both; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
    color: var(--text) !important;
}
.sidebar-brand {
    padding: 0.35rem 0.15rem 0.9rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}
.sidebar-brand .name {
    font-size: 1.25rem; font-weight: 700;
    color: var(--text) !important; letter-spacing: -0.02em;
}
.sidebar-brand .sub {
    font-size: 0.8rem; color: var(--text-2) !important; margin-top: 0.15rem;
}

.upload-shell { padding: 0.35rem 0.85rem 1rem; margin-bottom: 1rem; }
.upload-shell .upload-title {
    margin: 0.85rem 0 0.2rem; font-weight: 700;
    color: var(--text) !important; font-size: 1.05rem;
}
.upload-shell .upload-sub {
    margin: 0 0 0.75rem; color: var(--text-2) !important; font-size: 0.9rem;
}
[data-testid="stFileUploader"] section {
    background: linear-gradient(180deg, #FFFFFF 0%, #F7F7F1 100%) !important;
    border: 2px dashed #C5D2BC !important;
    border-radius: 16px !important;
    transition: all 0.28s ease;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--primary) !important;
    background: #EEF1E8 !important;
    transform: translateY(-2px);
    box-shadow: var(--shadow-hover);
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p { color: var(--text) !important; }

.doc-chip {
    display: flex; align-items: center; justify-content: space-between;
    gap: 12px; padding: 0.9rem 1.1rem; margin: 0.4rem 0 1rem;
    animation: fadeUp 0.45s ease both;
}
.doc-chip .meta strong { color: var(--text) !important; font-size: 0.98rem; }
.doc-chip .meta span { color: var(--text-2) !important; font-size: 0.85rem; }
.doc-chip .badge {
    flex-shrink: 0;
    background: rgba(113, 134, 107, 0.14);
    color: var(--primary) !important;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.28rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 700;
}

[data-testid="stAlert"] { animation: fadeUp 0.35s ease both; }

/* Hide sidebar on auth pages via collapsed default + empty — handled in page config per page */
.login-center .block-container { max-width: 560px; }

@media (max-width: 768px) {
    .brand-mark { font-size: 1.7rem; }
    .login-shell { padding: 1.35rem 1.2rem 1.15rem; margin: 0 0.25rem; }
    .stTextInput input { padding-right: 2.9rem !important; }
}
</style>
"""

BG_SCENE = """
<div class="bg-scene" aria-hidden="true">
  <div class="orb orb-a"></div>
  <div class="orb orb-b"></div>
  <div class="orb orb-c"></div>
  <div class="orb orb-d"></div>
  <div class="blob blob-1"></div>
  <div class="blob blob-2"></div>
  <div class="glow glow-1"></div>
  <div class="glow glow-2"></div>
  <div class="particle particle-a"></div>
  <div class="particle particle-b"></div>
  <div class="particle particle-c"></div>
</div>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(BG_SCENE, unsafe_allow_html=True)


def brand_header(tagline: str | None = None, show_logo: bool = True) -> None:
    tag = tagline or "Your Intelligent PDF Knowledge Space"
    logo = (
        '<div class="doc-logo" aria-hidden="true">'
        "<span></span><span></span><span></span></div>"
        if show_logo
        else ""
    )
    st.markdown(
        f"""
        <div class="brand-wrap">
            <div class="status-pill"><span class="dot"></span> Secure Access</div>
            {logo}
            <h1 class="brand-mark">DocuNest <span class="ai">AI</span></h1>
            <p class="brand-tagline">{tag}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="DocuNest AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="auto",
)
inject_css()

API_KEYS_URL = "https://platform.openai.com/api-keys"

# =============================================================================
# Routing helpers
# =============================================================================
st.session_state.setdefault("page", "login")
st.session_state.setdefault("api_key", "")


def go(page: str) -> None:
    st.session_state.page = page
    st.rerun()


def validate_key(key: str) -> tuple[bool, str]:
    """Make one cheap call to confirm the key works."""
    if not key.startswith("sk-"):
        return False, "That doesn't look like an OpenAI key (it should start with 'sk-')."
    try:
        OpenAI(api_key=key).models.list()
        return True, ""
    except AuthenticationError:
        return False, "OpenAI rejected this key. Check it and try again."
    except OpenAIError as e:
        return False, f"Couldn't verify the key: {e}"
    except Exception as e:
        return False, f"Connection problem: {e}"


def _fmt_size(num_bytes: int) -> str:
    kb = num_bytes / 1024
    if kb < 1024:
        return f"{kb:.1f} KB"
    return f"{kb / 1024:.2f} MB"


# =============================================================================
# Page: Login
# =============================================================================
def login_page() -> None:
    brand_header()
    _, mid, _ = st.columns([0.55, 2.0, 0.55])
    with mid:
        with st.form("login_form", clear_on_submit=False):
            st.markdown(
                """
                <h2 class="card-title" style="margin:0 0 0.3rem;font-size:1.4rem;font-weight:700;
                color:#28332B!important;-webkit-text-fill-color:#28332B!important">Welcome back</h2>
                <p class="card-desc" style="margin:0 0 1rem;color:#737D73!important;font-size:0.92rem;
                line-height:1.45">
                    Enter your OpenAI API key to unlock your intelligent PDF knowledge space.
                </p>
                """,
                unsafe_allow_html=True,
            )
            key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="Enter your OpenAI API key",
                value=os.getenv("OPENAI_API_KEY", ""),
                help="Stored only in this browser session. Never written to disk by DocuNest AI.",
            )
            submitted = st.form_submit_button("Continue  →", use_container_width=True)

        if submitted:
            key = key.strip()
            with st.spinner("Verifying your key…"):
                ok, err = validate_key(key)
            if ok:
                st.session_state.api_key = key
                st.toast("Welcome to DocuNest AI", icon="✅")
                go("chat")
            else:
                st.error(err)

        st.markdown(
            "<p class='login-link-row'>Need an API key?</p>",
            unsafe_allow_html=True,
        )
        if st.button("Create API key", use_container_width=True):
            go("signup")

        st.markdown(
            "<p class='login-foot'>Your key stays in this session and is used only "
            "to call OpenAI on your behalf. DocuNest AI never stores it on a server.</p>"
            "<p class='login-foot' style='margin-top:0.35rem!important;opacity:.85'>"
            "© DocuNest AI · Private PDF intelligence</p>",
            unsafe_allow_html=True,
        )


# =============================================================================
# Page: Sign up (how to create an API key)
# =============================================================================
def signup_page() -> None:
    brand_header(
        tagline="Create an OpenAI account and generate a key in about two minutes",
        show_logo=True,
    )
    _, mid, _ = st.columns([0.75, 1.6, 0.75])
    with mid:
        st.markdown(
            """
            <div class="step"><div class="num">1</div><div>
              <b>Create an OpenAI account</b><br>
              <span>Sign up at platform.openai.com with email, Google, or Microsoft.</span>
            </div></div>
            <div class="step"><div class="num">2</div><div>
              <b>Add billing credit</b><br>
              <span>API usage is pay-as-you-go. Add a small prepaid balance under Billing.</span>
            </div></div>
            <div class="step"><div class="num">3</div><div>
              <b>Create a secret key</b><br>
              <span>Open the API keys page, click <i>Create new secret key</i>, and name it.</span>
            </div></div>
            <div class="step"><div class="num">4</div><div>
              <b>Copy it and come back</b><br>
              <span>The key is shown only once. Copy it, then paste it on the login page.</span>
            </div></div>
            """,
            unsafe_allow_html=True,
        )
        st.link_button("Open OpenAI API keys page", API_KEYS_URL, use_container_width=True)
        st.write("")
        if st.button("I have my key — back to login", use_container_width=True):
            go("login")


# =============================================================================
# Page: Chat
# =============================================================================
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that answers questions strictly using the "
     "provided context from a PDF document.\n"
     "Rules:\n"
     "1. Answer ONLY from the context below.\n"
     "2. If the answer is not in the context, say "
     "\"I couldn't find that in the document.\"\n"
     "3. Cite the page number(s) you used, e.g. (p. 3).\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def build_vectorstore(pdf_bytes: bytes, size: int, overlap: int) -> FAISS:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        docs = PyPDFLoader(tmp_path).load()
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}" for d in docs
    )


def get_chain(vectorstore: FAISS, model: str, k: int):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model=model, temperature=0)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )
    return chain, retriever


def chat_page() -> None:
    os.environ["OPENAI_API_KEY"] = st.session_state.api_key

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="name">DocuNest AI</div>
                <div class="sub">PDF Knowledge Space</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Settings")
        model_name = st.selectbox(
            "Chat model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        )
        chunk_size = st.slider("Chunk size", 500, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
        top_k = st.slider("Retrieved chunks (k)", 2, 10, 4)
        st.divider()
        masked = f"sk-…{st.session_state.api_key[-4:]}"
        st.caption(f"Signed in with `{masked}`")
        if st.button("Log out", use_container_width=True):
            for k in ("api_key", "vectorstore", "file_sig", "messages", "pdf_name", "pdf_size"):
                st.session_state.pop(k, None)
            go("login")

    st.markdown(
        """
        <div class="brand-wrap" style="margin-bottom:1rem">
            <h1 class="brand-mark" style="font-size:1.75rem">DocuNest <span class="ai">AI</span></h1>
            <p class="brand-tagline">Upload a PDF, then ask questions — answers come only from your document</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="upload-shell">
            <p class="upload-title">Document upload</p>
            <p class="upload-sub">Drop a PDF to index it into your private knowledge space.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_pdf = st.file_uploader(
        "PDF file",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_pdf is not None:
        file_sig = (uploaded_pdf.name, uploaded_pdf.size, chunk_size, chunk_overlap)
        st.session_state.pdf_name = uploaded_pdf.name
        st.session_state.pdf_size = uploaded_pdf.size
        if st.session_state.get("file_sig") != file_sig:
            try:
                with st.spinner("Reading and indexing your PDF…"):
                    st.session_state.vectorstore = build_vectorstore(
                        uploaded_pdf.getvalue(), chunk_size, chunk_overlap
                    )
            except AuthenticationError:
                st.error("Your API key was rejected. Please log in again.")
                st.stop()
            except Exception as e:
                st.error(f"Couldn't index this PDF: {e}")
                st.stop()
            st.session_state.file_sig = file_sig
            st.session_state.messages = []
            st.toast(f"Indexed {uploaded_pdf.name}", icon="✅")

        st.markdown(
            f"""
            <div class="doc-chip">
                <div class="meta">
                    <strong>{uploaded_pdf.name}</strong><br>
                    <span>{_fmt_size(uploaded_pdf.size)} · ready for questions</span>
                </div>
                <div class="badge">INDEXED</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "vectorstore" not in st.session_state:
        st.markdown(
            "<div class='glass'>Upload a PDF above to begin your DocuNest session.</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        "<p class='section-sub' style='margin:0.2rem 0 0.8rem;font-weight:600;"
        "color:#28332B!important'>Conversation</p>",
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("messages", [])
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your PDF…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    chain, retriever = get_chain(st.session_state.vectorstore, model_name, top_k)

    with st.chat_message("assistant"):
        slot = st.empty()
        slot.markdown(
            "<div class='typing'><i></i><i></i><i></i></div>",
            unsafe_allow_html=True,
        )
        try:
            answer = chain.invoke(question)
        except AuthenticationError:
            slot.empty()
            st.error("Your API key was rejected. Please log in again.")
            return
        except Exception as e:
            slot.empty()
            st.error(f"Something went wrong: {e}")
            return
        slot.markdown(answer)

        with st.expander("Sources (retrieved chunks)"):
            for doc in retriever.invoke(question):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Page {page + 1 if isinstance(page, int) else page}**")
                st.text(doc.page_content[:500])
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})


# =============================================================================
# Router (chat is only reachable once a key is stored)
# =============================================================================
page = st.session_state.page
if page == "chat" and not st.session_state.api_key:
    page = "login"

{"login": login_page, "signup": signup_page, "chat": chat_page}[page]()
