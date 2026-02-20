import streamlit as st
import cv2
import numpy as np
import requests
import base64
import pandas as pd
from huggingface_hub import InferenceClient
from supabase import create_client

# --- 1. INITIALIZATION & SECURE CONNECTIVITY ---
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

# --- 2. FRONT-END: iOS OBSIDIAN DESIGN SYSTEM ---
st.set_page_config(page_title="TraidLive Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; font-family: -apple-system, sans-serif; }
    [data-testid="stMetricValue"] { font-size: 32px; font-weight: 700; color: #FFFFFF; }
    div[data-testid="stMetric"] { background: rgba(28, 28, 30, 0.8); backdrop-filter: blur(20px); border-radius: 18px; padding: 20px; border: 1px solid rgba(255, 255, 255, 0.1); }
    .stButton > button { background: linear-gradient(180deg, #0A84FF 0%, #007AFF 100%); color: white; border-radius: 14px; border: none; padding: 12px; font-weight: 600; width: 100%; transition: all 0.2s ease; }
    div.stTextInput > div > div > input { background-color: #1C1C1E; color: white; border-radius: 12px; }
    section[data-testid="stSidebar"] { background-color: #1C1C1E; border-right: 1px solid #3A3A3C; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ORIGINAL LOGIC (Untouched) ---

def fetch_market_valuation(card_name, grade_filter=""):
    token_url = "https://api.ebay.com/identity/v1/oauth2/token"
    auth_str = f"{EBAY_APP_ID}:{EBAY_CERT_ID}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    try:
        token_resp = requests.post(token_url, headers={"Authorization": f"Basic {encoded_auth}", "Content-Type": "application/x-www-form-urlencoded"}, data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"})
        token = token_resp.json().get("access_token")
        search_query = f"{card_name} {grade_filter} sold"
        query_encoded = requests.utils.quote(search_query)
        ebay_url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={query_encoded}&category_ids=212&limit=5"
        headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"}
        resp = requests.get(ebay_url, headers=headers)
        data = resp.json()
        items = data.get("itemSummaries", [])
        if not items: return 0.0
        prices = [float(item['price']['value']) for item in items if 'price' in item]
        return sum(prices) / len(prices)
    except: return 0.0

def auto_label_crops(crops):
    if not hf_client: return ["" for _ in crops]
    labels = []
    for crop in crops:
        try:
            _, buf = cv2.imencode(".jpg", crop)
            b64 = base64.b64encode(buf.tobytes()).decode()
            prompt = "Identify this card. Return ONLY: Year, Brand, Player, Card #. No prose."
            resp = hf_client.chat_completion(model=HF_MODEL, messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], max_tokens=50)
            labels.append(resp.choices[0].message.content.strip())
        except: labels.append("")
    return labels

def detect_cards(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    ratio = 1200.0 / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * ratio), 1200))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(edged, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 3000:
            x, y, w, h = cv2.boundingRect(approx)
            crops.append(img[y:y+h, x:x+w])
    return crops

# --- 4. THE PROFESSIONAL UI ---

st.title("TraidLive")
st.markdown("##### AI-Powered Asset Intel")

# KPI Summary Toolbar
k1, k2, k3 = st.columns(3)
k1.metric("Assets Syncing", "Active")
k2.metric("Market Data", "Real-Time")
k3.metric("System Health", "Optimal")

# Sidebar Controls
owner_id = st.sidebar.text_input("Customer ID", value="nbult99")
st.sidebar.divider()
st.sidebar.caption("v2.0 obsidian.ios.dark")

# CAMERA VS GALLERY TOGGLE
source = st.radio("Asset Source", ["Gallery", "Camera"], horizontal=True)

if source == "Camera":
    uploaded_file = st.camera_input("Snap a photo of your cards")
else:
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with st.spinner("Analyzing Assets..."):
        uploaded_file.seek(0)
        asset_crops = detect_cards(uploaded_file)
    
    if asset_crops:
        col_ai, col_commit_all = st.columns(2)
        with col_ai:
            if st.button("AI Batch Identification"):
                with st.spinner("Processing..."):
                    st.session_state['suggestions'] = auto_label_crops(asset_crops)

        with col_commit_all:
            if 'suggestions' in st.session_state:
                if st.button("Commit All to Database"):
                    with st.spinner("Synchronizing..."):
                        for i, name in enumerate(st.session_state['suggestions']):
                            p_psa = fetch_market_valuation(name, "PSA 10")
                            p_raw = fetch_market_valuation(name, "Ungraded")
                            supabase.table("inventory").insert({"card_name": name, "psa_10_price": p_psa, "ungraded_price": p_raw, "owner": owner_id}).execute()
                        st.success(f"Successfully committed {len(st.session_state['suggestions'])} assets.")

        if 'suggestions' in st.session_state:
            st.divider()
            grid = st.columns(4)
            for i, crop in enumerate(asset_crops):
                with grid[i % 4]:
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.session_state['suggestions'][i] = st.text_input(f"ID {i+1}", value=st.session_state['suggestions'][i], key=f"inp_{i}")
                    
                    if st.button(f"Commit {i+1}", key=f"btn_{i}"):
                        p_psa = fetch_market_valuation(st.session_state['suggestions'][i], "PSA 10")
                        p_raw = fetch_market_valuation(st.session_state['suggestions'][i], "Ungraded")
                        supabase.table("inventory").insert({"card_name": st.session_state['suggestions'][i], "psa_10_price": p_psa, "ungraded_price": p_raw, "owner": owner_id}).execute()
                        st.toast(f"Synced {st.session_state['suggestions'][i]}")
    else:
        st.warning("No card contours detected.")

# Ledger Section
st.divider()
st.subheader("Inventory Vault")
if st.button("Refresh Records"):
    try:
        response = supabase.table("inventory").select("*").eq("owner", owner_id).order("created_at", desc=True).execute()
        if response.data:
            df = pd.DataFrame(response.data)
            df = df[['card_name', 'ungraded_price', 'psa_10_price', 'created_at']]
            df.columns = ['Asset Name', 'Raw Market', 'PSA 10 Floor', 'Sync Date']
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Sync failed: {str(e)}")