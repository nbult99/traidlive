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

# --- 2. ROBINHOOD DESIGN SYSTEM ---
st.set_page_config(page_title="TraidLive", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; font-family: -apple-system, system-ui, sans-serif; }
    .stButton > button { background-color: #00C805 !important; color: #000000 !important; border-radius: 24px !important; border: none !important; font-weight: 700 !important; transition: 0.2s ease; }
    .nav-container { position: fixed; right: 40px; top: 40px; z-index: 100; text-align: right; }
    .nav-item { color: rgba(255, 255, 255, 0.4); text-decoration: none; font-size: 16px; font-weight: 600; margin-bottom: 20px; display: block; transition: 0.3s; }
    .nav-item:hover { color: #00C805; }
    .sold-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #1E2124; }
    .sold-price { color: #00C805; font-weight: 600; font-family: monospace; }
    [data-testid="stSidebar"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. REFINED VISION ENGINE (ANTI-GHOSTING) ---
def detect_cards(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 1. Standardize size for consistent detection
    ratio = 1200.0 / img.shape[0]
    img_work = cv2.resize(img, (int(img.shape[1] * ratio), 1200))
    
    # 2. Advanced Pre-processing to kill shadows
    gray = cv2.cvtColor(img_work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    # Use Otsu's thresholding to separate card from background
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crops = []
    # Sort by area and take only the largest ones
    sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in sorted_cnts:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # --- REFINEMENT RULES ---
        # Rule 1: Area must be significant (Filters out "ghost" flecks)
        if area < 40000: continue 
        
        # Rule 2: Must have 4 corners (Filters out non-rectangular shadows)
        # Rule 3: Must not exceed 8 total cards
        if len(approx) >= 4 and len(crops) < 8:
            x, y, w, h = cv2.boundingRect(approx)
            
            # Rule 4: Aspect Ratio Check (Cards are tall rectangles)
            aspect_ratio = float(w)/h
            if 0.5 < aspect_ratio < 0.9 or 1.1 < aspect_ratio < 1.8:
                # Add 15px padding so AI sees the card edges better
                pad = 15
                y1, y2 = max(0, y-pad), min(img_work.shape[0], y+h+pad)
                x1, x2 = max(0, x-pad), min(img_work.shape[1], x+w+pad)
                crops.append(img_work[y1:y2, x1:x2])
                
    return crops

# --- 4. EBAY & AI MODULES (RETAINED) ---
def fetch_market_valuation(card_name, grade_filter=""):
    token_url = "https://api.ebay.com/identity/v1/oauth2/token"
    auth_str = f"{EBAY_APP_ID}:{EBAY_CERT_ID}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    try:
        token_resp = requests.post(token_url, headers={"Authorization": f"Basic {encoded_auth}"}, data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"})
        token = token_resp.json().get("access_token")
        search_query = f"{card_name} {grade_filter} sold" if grade_filter != "Ungraded" else f"{card_name} -PSA -BGS -SGC -CGC -graded sold"
        ebay_url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={requests.utils.quote(search_query)}&category_ids=212&limit=20"
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

# --- 5. INTERFACE ---
st.markdown("""<div class="nav-container">
    <a class="nav-item" href="/?page=Home">Home</a>
    <a class="nav-item" href="/?page=Inventory">Inventory</a>
    <a class="nav-item" href="/?page=Search">Search</a>
    <a class="nav-item" href="/?page=Profile">Profile</a>
</div>""", unsafe_allow_html=True)

page = st.query_params.get("page", "Home")

if page == "Home":
    st.title("TraidLive")
    st.markdown("<h3 style='color: #8E8E93;'>Welcome! Scan up to eight cards with one photo.</h3>", unsafe_allow_html=True)
    
    source = st.radio("Scanner", ["Camera", "Gallery"], horizontal=True)
    img_input = st.camera_input("Active") if source == "Camera" else st.file_uploader("Drop Image")

    if img_input:
        with st.spinner("Cleaning image & identifying assets..."):
            img_input.seek(0)
            asset_crops = detect_cards(img_input)
            
            if asset_crops:
                st.write(f"Detected {len(asset_crops)} valid assets.")
                if st.button("AI BATCH IDENTIFY"):
                    st.session_state['results'] = auto_label_crops(asset_crops)
        
        if 'results' in st.session_state:
            st.divider()
            grid = st.columns(4)
            for i, name in enumerate(st.session_state['results']):
                with grid[i % 4]:
                    st.image(cv2.cvtColor(asset_crops[i], cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.session_state['results'][i] = st.text_input(f"ID {i+1}", value=name, key=f"n_{i}")
                    
                    if st.button(f"CHECK MARKET {i+1}", key=f"p_{i}"):
                        html_rows = ""
                        for label, gr_query in [("PSA 10", "PSA 10"), ("PSA 9", "PSA 9"), ("PSA 8", "PSA 8"), ("RAW", "Ungraded")]:
                            val, points = fetch_market_valuation(name, gr_query)
                            if points:
                                list_html = "".join([f"<div class='sold-row'><span class='sold-title'>{p['title']}</span><span class='sold-price'>${p['price']:,.2f}</span><a href='{p['url']}' target='_blank'>Link</a></div>" for p in points])
                                html_rows += f"<tr><td>{label}</td><td style='text-align:right;'><details><summary style='color:#00C805;'>${val:,.2f} ▼</summary><div style='background:#151516; padding:10px; border-radius:8px; border:1px solid #3A3A3C; text-align:left; margin-top:10px;'>{list_html}</div></details></td></tr>"
                        st.markdown(f"<div style='background:#1E2124; border-radius:12px; padding:10px; border:1px solid #30363D;'><table style='width:100%;'>{html_rows}</table></div>", unsafe_allow_html=True)

elif page == "Inventory": st.title("Vault")