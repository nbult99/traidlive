import streamlit as st
import cv2
import numpy as np
import requests
import base64
import pandas as pd
from huggingface_hub import InferenceClient
from supabase import create_client

# --- 1. INITIALIZATION (BACKEND UNTOUCHED) ---
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
EBAY_APP_ID = st.secrets.get("EBAY_APP_ID", "")
EBAY_CERT_ID = st.secrets.get("EBAY_CERT_ID", "")

@st.cache_resource
def init_connections():
    s_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    h_client = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else None
    return s_client, h_client

supabase, hf_client = init_connections()
HF_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# --- 2. ROBINHOOD DESIGN SYSTEM (ADAPTIVE) ---
st.set_page_config(page_title="TraidLive", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Adaptive Dark/Light Mode & Robinhood Green */
    :root {
        --rh-green: #00C805;
        --rh-black: #000000;
        --rh-card: #1E2124;
    }

    /* Target Robinhood Green for buttons */
    .stButton > button {
        background-color: var(--rh-green) !important;
        color: black !important;
        border-radius: 24px !important;
        border: none !important;
        font-weight: 700 !important;
        width: 100%;
        transition: transform 0.2s ease;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        background-color: #00E606 !important;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: var(--rh-card);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 15px;
    }
    
    /* Navigation Bar Simulation */
    .nav-text {
        font-family: -apple-system, system-ui, sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 640px) {
        .stActionButton { display: none; }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS (RETAINED FROM STABLE BUILD) ---
def fetch_market_valuation(card_name, grade_filter=""):
    token_url = "https://api.ebay.com/identity/v1/oauth2/token"
    auth_str = f"{EBAY_APP_ID}:{EBAY_CERT_ID}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    try:
        token_resp = requests.post(token_url, headers={"Authorization": f"Basic {encoded_auth}"}, data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"})
        token = token_resp.json().get("access_token")
        search_query = f"{card_name} {grade_filter} sold"
        ebay_url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={requests.utils.quote(search_query)}&category_ids=212&limit=10"
        resp = requests.get(ebay_url, headers={"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"})
        items = resp.json().get("itemSummaries", [])
        if not items: return 0.0, []
        points = [{"title": i['title'], "price": float(i['price']['value']), "url": i.get('itemWebUrl', '#')} for i in items if 'price' in i]
        return (sum(p['price'] for p in points) / len(points)), points
    except: return 0.0, []

def auto_label_crops(crops):
    labels = []
    for crop in crops:
        _, buf = cv2.imencode(".jpg", crop)
        b64 = base64.b64encode(buf.tobytes()).decode()
        resp = hf_client.chat_completion(model=HF_MODEL, messages=[{"role": "user", "content": [{"type": "text", "text": "Identify this card: Year, Brand, Player, Card #. No prose."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], max_tokens=50)
        labels.append(resp.choices[0].message.content.strip())
    return labels

def detect_cards(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    ratio = 1200.0 / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * ratio), 1200))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 30, 150)
    contours, _ = cv2.findContours(cv2.dilate(edged, None, iterations=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        if cv2.contourArea(cnt) > 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            crops.append(img[y:y+h, x:x+w])
    return crops

# --- 4. MULTI-PAGE NAVIGATION ---
page = st.sidebar.selectbox("Navigate", ["Home", "Search & Identify", "Inventory", "Profile"])

# --- PAGE: HOME ---
if page == "Home":
    st.title("Welcome to TraidLive")
    st.subheader("Your AI-Powered Trading Floor")
    st.write("Track market movement, verify assets, and manage your vault with high-frequency precision.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Market Status", "OPEN", delta="Active")
    col2.metric("Top Mover", "Tom Brady RC", delta="12.4%", delta_color="normal")
    col3.metric("Vault Value", "$24,502.10", delta="+$420.00")

# --- PAGE: SEARCH & IDENTIFY ---
elif page == "Search & Identify":
    st.title("Identify & Value")
    input_method = st.radio("Input Source", ["Camera", "Upload"], horizontal=True)
    
    if input_method == "Camera":
        uploaded_file = st.camera_input("Snap Card")
    else:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        crops = detect_cards(uploaded_file)
        if st.button("AI Identify Batch"):
            st.session_state['suggestions'] = auto_label_crops(crops)
            st.rerun()
            
        if 'suggestions' in st.session_state:
            cols = st.columns(4)
            for i, name in enumerate(st.session_state['suggestions']):
                with cols[i % 4]:
                    st.text_input(f"Asset {i+1}", value=name, key=f"inp_{i}")
                    if st.button(f"Analyze {i+1}", key=f"btn_{i}"):
                        val, pts = fetch_market_valuation(name, "PSA 10")
                        st.success(f"Avg: ${val:,.2f}")

# --- PAGE: INVENTORY ---
elif page == "Inventory":
    st.title("Asset Vault")
    st.write("Your synchronized historical ledger.")
    if st.button("Refresh Vault"):
        res = supabase.table("inventory").select("*").eq("owner", owner_id).execute()
        st.dataframe(pd.DataFrame(res.data), use_container_width=True)

# --- PAGE: PROFILE ---
elif page == "Profile":
    st.title("Collector Profile")
    st.write(f"Account: **{owner_id}**")
    st.button("Account Settings")
    st.button("API Integration")