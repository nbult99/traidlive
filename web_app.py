import streamlit as st
import cv2
import numpy as np
import requests
import base64
import pandas as pd
import re
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
    .stButton > button { background-color: #00C805 !important; color: #000000 !important; border-radius: 24px !important; border: none !important; font-weight: 700 !important; transition: 0.2s ease; width: 100%; }
    .stButton > button:hover { background-color: #00E606 !important; transform: translateY(-1px); }
    .nav-container { position: fixed; right: 40px; top: 40px; z-index: 100; text-align: right; }
    .nav-item { color: rgba(255, 255, 255, 0.4); text-decoration: none; font-size: 16px; font-weight: 600; margin-bottom: 20px; display: block; transition: 0.3s; }
    .nav-item:hover { color: #00C805; }
    details > summary { list-style: none; outline: none; cursor: pointer; }
    .sold-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #1E2124; }
    .sold-price { color: #00C805; font-weight: 600; font-family: monospace; }
    [data-testid="stSidebar"] { display: none; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. CORE ENGINES ---
def fetch_market_valuation(card_id_data, grade_filter=""):
    token_url = "https://api.ebay.com/identity/v1/oauth2/token"
    auth_str = f"{EBAY_APP_ID}:{EBAY_CERT_ID}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    
    year = str(card_id_data.get('year', '')).strip()
    brand = str(card_id_data.get('brand', '')).strip().upper()
    player = str(card_id_data.get('player', '')).strip().upper()
    num = str(card_id_data.get('card_num', '')).strip()

    try:
        token_resp = requests.post(token_url, headers={"Authorization": f"Basic {encoded_auth}"}, data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"})
        token = token_resp.json().get("access_token")
        
        search_query = f"{year} {brand} {player} {num} {grade_filter} sold -reprint -rp"
        ebay_url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={requests.utils.quote(search_query)}&category_ids=212&limit=40"
        
        resp = requests.get(ebay_url, headers={"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"})
        items = resp.json().get("itemSummaries", [])
        
        points = []
        for item in items:
            title = item.get('title', '').upper()
            has_year = year in title
            has_num = num in title
            has_player = all(part in title for part in player.split() if len(part) > 2)
            
            if has_year and has_num and has_player:
                points.append({"title": item['title'], "price": float(item['price']['value']), "url": item.get('itemWebUrl', '#')})
            if len(points) >= 10: break
                
        if not points: return 0.0, []
        return (sum(p['price'] for p in points) / len(points)), points
    except: return 0.0, []

def auto_label_crops(crops):
    results = []
    for crop in crops:
        try:
            _, buf = cv2.imencode(".jpg", crop)
            b64 = base64.b64encode(buf.tobytes()).decode()
            prompt = "Sports card ID: Year | Brand | Player | Card #. Example: 2000 | Bowman | Tom Brady | 236"
            resp = hf_client.chat_completion(model=HF_MODEL, messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], max_tokens=60)
            parts = [p.strip() for p in resp.choices[0].message.content.strip().split('|')]
            if len(parts) >= 4:
                results.append({"full_name": f"{parts[0]} {parts[1]} {parts[2]} #{parts[3]}", "year": parts[0], "brand": parts[1], "player": parts[2], "card_num": parts[3]})
            else: results.append({"full_name": "Check ID", "year": "", "brand": "", "player": "", "card_num": ""})
        except: results.append({"full_name": "Error", "year": "", "brand": "", "player": "", "card_num": ""})
    return results

def detect_cards(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    ratio = 1200.0 / img.shape[0]
    img_work = cv2.resize(img, (int(img.shape[1] * ratio), 1200))
    gray = cv2.cvtColor(img_work, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 150)
    dilated = cv2.dilate(edged, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        if cv2.contourArea(cnt) > 20000:
            x, y, w, h = cv2.boundingRect(cnt)
            pad = 10
            y1, y2 = max(0, y-pad), min(img_work.shape[0], y+h+pad)
            x1, x2 = max(0, x-pad), min(img_work.shape[1], x+w+pad)
            crops.append(img_work[y1:y2, x1:x2])
    return crops

# --- 4. NAVIGATION ---
page = st.query_params.get("page", "Home")

st.markdown("""<div class="nav-container">
    <a class="nav-item" href="/?page=Home">Home</a>
    <a class="nav-item" href="/?page=Inventory">Inventory</a>
    <a class="nav-item" href="/?page=Search">Search</a>
    <a class="nav-item" href="/?page=Profile">Profile</a>
</div>""", unsafe_allow_html=True)

# --- 5. HOME PAGE ---
if page == "Home":
    st.title("TraidLive")
    st.markdown("<h3 style='color: #8E8E93;'>Terminal v1.0 | Real-time Market Intelligence</h3>", unsafe_allow_html=True)

    # Persistent Storage for scanner results
    if 'results' not in st.session_state: st.session_state['results'] = []
    if 'crops' not in st.session_state: st.session_state['crops'] = []

    # Scanner Block
    source = st.radio("Source", ["Camera", "Gallery"], horizontal=True)
    img_input = st.camera_input("Scanner") if source == "Camera" else st.file_uploader("Upload Image")

    if img_input:
        img_input.seek(0)
        asset_crops = detect_cards(img_input)
        
        if asset_crops and st.button("AI BATCH IDENTIFY", use_container_width=True):
            with st.spinner("Decoding asset identity..."):
                st.session_state['results'] = auto_label_crops(asset_crops)
                st.session_state['crops'] = asset_crops
    
    # Render the identified grid
    if st.session_state['results']:
        st.divider()
        grid = st.columns(4)
        for i, data in enumerate(st.session_state['results']):
            with grid[i % 4]:
                # Display the crop
                st.image(cv2.cvtColor(st.session_state['crops'][i], cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Manual override for identity
                display_name = st.text_input(f"Asset ID {i+1}", value=data['full_name'], key=f"n_{i}")
                
                # ACTION: Check Market
                if st.button(f"CHECK MARKET", key=f"p_{i}"):
                    # Logic to re-parse the string if user edited it
                    p = display_name.replace('#', '').split()
                    s_data = data if len(p) < 4 else {"year": p[0], "brand": p[1], "player": ' '.join(p[2:-1]), "card_num": p[-1]}
                    
                    with st.spinner("Polling eBay..."):
                        html_r = ""
                        for l, g in [("PSA 10", "PSA 10"), ("PSA 9", "PSA 9"), ("PSA 8", "PSA 8"), ("RAW", "Ungraded")]:
                            v, pts = fetch_market_valuation(s_data, g)
                            if pts:
                                l_html = "".join([f"<div class='sold-row'><span class='sold-title'>{x['title']}</span><span class='sold-price'>${x['price']:,.2f}</span><a href='{x['url']}' target='_blank' style='color:#0A84FF;'>Sold</a></div>" for x in pts])
                                html_r += f"<tr style='border-bottom:1px solid #2C2C2E;'><td style='padding:10px 5px;'><strong>{l}</strong></td><td style='padding:10px 5px; text-align: right;'><details><summary style='color:#00C805; font-weight:700;'>${v:,.2f} ▼</summary><div style='background:#151516; padding:10px; border-radius:8px; border:1px solid #3A3A3C; text-align:left; margin-top:10px;'>{l_html}</div></details></td></tr>"
                        
                        if html_r:
                            st.markdown(f"<div style='background:#1E2124; border-radius:12px; padding:15px; border:1px solid #30363D; margin-top:10px;'><table style='width:100%; border-collapse:collapse;'>{html_r}</table></div>", unsafe_allow_html=True)
                        else:
                            st.warning("No clinical matches found in current market.")

elif page == "Inventory": st.title("Vault")