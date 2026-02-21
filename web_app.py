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

# --- 3. MARKET DATA ENGINE (SMART GRADE FILTER) ---

def fetch_market_valuation(card_id_data, grade_filter=""):
    """
    SMART FILTER: Strict on Grade, Flexible on Metadata.
    """
    token_url = "https://api.ebay.com/identity/v1/oauth2/token"
    auth_str = f"{EBAY_APP_ID}:{EBAY_CERT_ID}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    
    target_year = str(card_id_data.get('year', '')).strip()
    target_brand = str(card_id_data.get('brand', '')).strip().upper()
    target_player = str(card_id_data.get('player', '')).strip().upper()
    target_num = str(card_id_data.get('num', '')).strip()

    try:
        token_resp = requests.post(token_url, headers={"Authorization": f"Basic {encoded_auth}"}, data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"})
        token = token_resp.json().get("access_token")
        
        # Build query (Broad search to capture lazy titles)
        search_query = f"{target_year} {target_brand} {target_player} {target_num} {grade_filter} sold -reprint"
        if grade_filter == "Ungraded":
            search_query = f"{target_year} {target_brand} {target_player} {num} sold -PSA -BGS -SGC -CGC -graded -reprint"
            
        ebay_url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={requests.utils.quote(search_query)}&category_ids=212&limit=25"
        resp = requests.get(ebay_url, headers={"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"})
        items = resp.json().get("itemSummaries", [])
        
        points = []
        for item in items:
            title = item.get('title', '').upper()
            
            # --- THE GRADE FIREWALL ---
            # If we want a PSA 10, we EXCLUDE any title that explicitly mentions 9, 8, 7, 6, 5 etc.
            if "PSA" in grade_filter:
                grade_val = grade_filter.split()[-1] # Gets '10' from 'PSA 10'
                other_grades = ["PSA 10", "PSA 9", "PSA 8", "PSA 7", "PSA 6", "PSA 5"]
                # If title mentions a DIFFERENT PSA grade than the one we are currently looking for, skip it.
                if any(g in title for g in other_grades if g != grade_filter):
                    continue
            
            # --- THE FUZZY DATA CHECK ---
            # As long as the PLAYER is in the title, we accept it (Flexible on year/num/brand)
            has_player = all(part in title for part in target_player.split() if len(part) > 2)
            
            if has_player:
                points.append({"title": item['title'], "price": float(item['price']['value']), "url": item.get('itemWebUrl', '#')})
            
            if len(points) >= 10: break
                
        if not points: return 0.0, []
        avg = sum(p['price'] for p in points) / len(points)
        return avg, points
    except: return 0.0, []

def auto_label_crops(crops):
    labels = []
    for crop in crops:
        try:
            _, buf = cv2.imencode(".jpg", crop)
            b64 = base64.b64encode(buf.tobytes()).decode()
            prompt = "Precisely identify: Year | Brand | Player | Card #. Example: 2000 | Bowman | Tom Brady | 236"
            resp = hf_client.chat_completion(model=HF_MODEL, messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], max_tokens=50)
            raw = resp.choices[0].message.content.strip()
            parts = [p.strip() for p in raw.split('|')]
            if len(parts) >= 4:
                labels.append({"full": f"{parts[0]} {parts[1]} {parts[2]} #{parts[3]}", "year": parts[0], "brand": parts[1], "player": parts[2], "num": parts[3]})
            else: labels.append({"full": "ID Failed", "year": "", "brand": "", "player": "", "num": ""})
        except: labels.append({"full": "Error", "year": "", "brand": "", "player": "", "num": ""})
    return labels

def detect_cards(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    ratio = 1200.0 / img.shape[0]
    img_work = cv2.resize(img, (int(img.shape[1] * ratio), 1200))
    gray = cv2.cvtColor(img_work, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 150)
    dilated = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
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

def display_pricing_table(card_data):
    html_rows = ""
    for label, gr_query in [("PSA 10", "PSA 10"), ("PSA 9", "PSA 9"), ("PSA 8", "PSA 8"), ("RAW", "Ungraded")]:
        val, points = fetch_market_valuation(card_data, gr_query)
        if points:
            list_html = "".join([f"<div class='sold-row'><span class='sold-title'>{p['title']}</span><span class='sold-price'>${p['price']:,.2f}</span><a href='{p['url']}' target='_blank' style='color:#0A84FF; text-decoration:none;'>Link</a></div>" for p in points])
            html_rows += f"<tr><td style='padding:8px 0;'>{label}</td><td style='text-align:right;'><details><summary style='color:#00C805; font-weight:bold;'>${val:,.2f} ▼</summary><div style='background:#151516; padding:10px; border-radius:8px; border:1px solid #3A3A3C; text-align:left; margin-top:10px;'>{list_html}</div></details></td></tr>"
    if html_rows:
        st.markdown(f"<div style='background:#1E2124; border-radius:12px; padding:15px; border:1px solid #30363D;'><table style='width:100%;'>{html_rows}</table></div>", unsafe_allow_html=True)
    else:
        st.warning("No listings found.")

# --- 4. THE HYBRID INTERFACE ---

st.markdown("""<div class="nav-container">
    <a class="nav-item" href="/?page=Home">Home</a>
    <a class="nav-item" href="/?page=Inventory">Inventory</a>
    <a class="nav-item" href="/?page=Search">Search</a>
    <a class="nav-item" href="/?page=Profile">Profile</a>
</div>""", unsafe_allow_html=True)

page = st.query_params.get("page", "Home")

if page == "Home":
    st.title("TraidLive")
    st.markdown("<h3 style='color: #8E8E93;'>Terminal v1.2 | Smart Grade Sorting Active</h3>", unsafe_allow_html=True)
    
    search_q = st.text_input("Quick Lookup", placeholder="Year Brand Player Card#")
    if search_q and st.button("GET MARKET DATA"):
        p = search_q.replace('#', '').split()
        if len(p) >= 4:
            display_pricing_table({"year": p[0], "brand": p[1], "player": ' '.join(p[2:-1]), "num": p[-1]})

    st.divider()

    input_type = st.radio("Input Method", ["Search with Text", "Upload Image"], horizontal=True)

    if input_type == "Upload Image":
        upload_option = st.radio("Source:", ["Browse Files", "Use Camera"], horizontal=True)
        img_input = st.camera_input("Scanner") if upload_option == "Use Camera" else st.file_uploader("Select Image", type=['jpg','jpeg','png'])

        if img_input:
            img_input.seek(0)
            asset_crops = detect_cards(img_input)
            if asset_crops:
                if st.button("AI BATCH IDENTIFY"):
                    st.session_state['scan_results'] = auto_label_crops(asset_crops)
                    st.session_state['scan_crops'] = asset_crops
            
            if 'scan_results' in st.session_state:
                grid = st.columns(4)
                for i, card_data in enumerate(st.session_state['scan_results']):
                    with grid[i % 4]:
                        st.image(cv2.cvtColor(st.session_state['scan_crops'][i], cv2.COLOR_BGR2RGB), use_container_width=True)
                        display_name = st.text_input(f"Asset {i+1}", value=card_data['full'], key=f"sn_{i}")
                        if st.button(f"Analyze {i+1}", key=f"sp_{i}"):
                            p = display_name.replace('#', '').split()
                            final_data = card_data if len(p) < 4 else {"year": p[0], "brand": p[1], "player": ' '.join(p[2:-1]), "num": p[-1]}
                            display_pricing_table(final_data)

elif page == "Inventory": st.title("Vault")