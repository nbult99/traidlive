import streamlit as st
import cv2
import numpy as np
import requests
import base64
import pandas as pd
from huggingface_hub import InferenceClient
from supabase import create_client

# --- 1. INITIALIZATION ---
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

# --- 2. ROBINHOOD DESIGN SYSTEM (FIXED NAV) ---
st.set_page_config(page_title="TraidLive", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Robinhood Trading Terminal Aesthetic */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
        font-family: -apple-system, system-ui, sans-serif;
    }
    
    /* Neon Green Robinhood Accents */
    .stButton > button {
        background-color: #00C805 !important;
        color: #000000 !important;
        border-radius: 24px !important;
        border: none !important;
        font-weight: 700 !important;
        transition: 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #00E606 !important;
        transform: translateY(-1px);
    }

    /* Transparent Right-Side Navigation */
    .nav-container {
        position: fixed;
        right: 40px;
        top: 40px;
        z-index: 100;
        text-align: right;
    }
    .nav-item {
        color: rgba(255, 255, 255, 0.4);
        text-decoration: none;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 20px;
        display: block;
        transition: 0.3s;
    }
    .nav-item:hover {
        color: #00C805;
    }

    /* Ticker-style Table Styling */
    .sold-row { border-bottom: 1px solid #1E2124; padding: 10px 0; }
    .price-text { color: #00C805; font-weight: 700; font-family: monospace; }
    
    /* Hide default Streamlit sidebar and clutter */
    [data-testid="stSidebar"] { display: none; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. THE BACKEND ENGINE ---
def fetch_market_valuation(card_name, grade_filter=""):
    token_url = "https://api.ebay.com/identity/v1/oauth2/token"
    auth_str = f"{EBAY_APP_ID}:{EBAY_CERT_ID}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    try:
        token_resp = requests.post(token_url, headers={"Authorization": f"Basic {encoded_auth}"}, data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"})
        token = token_resp.json().get("access_token")
        
        # negative keywords for raw search
        if grade_filter == "Ungraded":
            search_query = f"{card_name} -PSA -BGS -SGC -CGC -graded sold"
        else:
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
        resp = hf_client.chat_completion(model=HF_MODEL, messages=[{"role": "user", "content": [{"type": "text", "text": "Identify this card. Year, Brand, Player, Card #. No prose."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], max_tokens=50)
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

# --- 4. TERMINAL UI STRUCTURE ---

# Right-Side Navigation
st.markdown("""
    <div class="nav-container">
        <a class="nav-item" href="/?page=Home">Home</a>
        <a class="nav-item" href="/?page=Inventory">Inventory</a>
        <a class="nav-item" href="/?page=Search">Search</a>
        <a class="nav-item" href="/?page=Profile">Profile</a>
    </div>
    """, unsafe_allow_html=True)

# Using Query Params to handle the "Pages" since we hid the sidebar
params = st.query_params
current_page = params.get("page", "Home")

if current_page == "Home":
    st.title("TraidLive")
    st.markdown("<h3 style='color: #8E8E93;'>Welcome! Scan up to eight cards with one photo to find current listings.</h3>", unsafe_allow_html=True)
    
    # Hero Scanner
    source = st.radio("Select Source", ["Camera", "Gallery"], horizontal=True)
    if source == "Camera":
        img_input = st.camera_input("Scanner Active")
    else:
        img_input = st.file_uploader("Drop image here")

    if img_input:
        with st.spinner("Processing Assets..."):
            img_input.seek(0)
            crops = detect_cards(img_input)
            
            if st.button("AI BATCH IDENTIFY"):
                st.session_state['results'] = auto_label_crops(crops)
        
        if 'results' in st.session_state:
            st.divider()
            cols = st.columns(4)
            for i, name in enumerate(st.session_state['results']):
                with cols[i % 4]:
                    st.image(cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.session_state['results'][i] = st.text_input(f"ID {i+1}", value=name, key=f"n_{i}")
                    
                    if st.button(f"CHECK MARKET {i+1}", key=f"p_{i}"):
                        for grade in ["PSA 10", "PSA 9", "PSA 8", "Ungraded"]:
                            avg, pts = fetch_market_valuation(name, grade)
                            st.markdown(f"**{grade}**: <span class='price-text'>${avg:,.2f}</span>", unsafe_allow_html=True)

elif current_page == "Inventory":
    st.title("Vault")
    st.info("Inventory details will populate here.")

elif current_page == "Search":
    st.title("Search Market")
    manual_q = st.text_input("Manual Search (eBay Data)")

elif current_page == "Profile":
    st.title("User Settings")
    st.write(f"Logged in as: {owner_id}")