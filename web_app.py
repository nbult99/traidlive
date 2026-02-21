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
    
    /* Neon Green Robinhood Accents */
    .stButton > button { background-color: #00C805 !important; color: #000000 !important; border-radius: 24px !important; border: none !important; font-weight: 700 !important; transition: 0.2s ease; }
    .stButton > button:hover { background-color: #00E606 !important; transform: translateY(-1px); }

    /* Transparent Right-Side Navigation */
    .nav-container { position: fixed; right: 40px; top: 40px; z-index: 100; text-align: right; }
    .nav-item { color: rgba(255, 255, 255, 0.4); text-decoration: none; font-size: 16px; font-weight: 600; margin-bottom: 20px; display: block; transition: 0.3s; }
    .nav-item:hover { color: #00C805; }

    /* Interactive Table Styling */
    details > summary { list-style: none; outline: none; cursor: pointer; }
    details > summary::-webkit-details-marker { display: none; }
    .audit-header { color: #8E8E93; font-size: 11px; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.5px; font-weight: bold; }
    .sold-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #1E2124; }
    .sold-title { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-right: 10px; color: #D1D1D6; font-size: 13px; }
    .sold-price { color: #00C805; font-weight: 600; margin-right: 10px; font-size: 13px; font-family: monospace; }
    .sold-link { color: #0A84FF; text-decoration: none; font-size: 12px; font-weight: 500; }
    
    /* Hide default Streamlit clutter */
    [data-testid="stSidebar"] { display: none; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. MARKET DATA ENGINE (STRICT) ---
def fetch_market_valuation(card_name, grade_filter=""):
    token_url = "https://api.ebay.com/identity/v1/oauth2/token"
    auth_str = f"{EBAY_APP_ID}:{EBAY_CERT_ID}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    try:
        token_resp = requests.post(token_url, headers={"Authorization": f"Basic {encoded_auth}"}, data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"})
        token = token_resp.json().get("access_token")
        
        if grade_filter == "Ungraded":
            search_query = f"{card_name} -PSA -BGS -SGC -CGC -graded sold"
        else:
            search_query = f"{card_name} {grade_filter} sold"
            
        ebay_url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={requests.utils.quote(search_query)}&category_ids=212&limit=20"
        headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"}
        
        resp = requests.get(ebay_url, headers=headers)
        items = resp.json().get("itemSummaries", [])
        if not items: return 0.0, []
        
        points = []
        for item in items:
            if 'price' in item:
                title = item.get('title', 'Unknown Item')
                is_valid = True
                
                if grade_filter == "Ungraded":
                    if any(g in title.upper() for g in ["PSA", "BGS", "SGC", "CGC"]): is_valid = False
                elif "PSA" in grade_filter:
                    grade_num = grade_filter.replace("PSA", "").replace("(", "").replace(")", "").strip()
                    if grade_filter == "PSA (1, 2, 3, 4, 5, 6)":
                         if not any(str(i) in title for i in range(1, 7)): is_valid = False
                    elif grade_num not in title: is_valid = False

                if is_valid:
                    points.append({"title": title, "price": float(item['price']['value']), "url": item.get('itemWebUrl', '#')})
            if len(points) >= 10: break
                
        if not points: return 0.0, []
        avg = sum(p['price'] for p in points) / len(points)
        return avg, points
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

# --- 4. TERMINAL UI ---

# Navigation
st.markdown("""<div class="nav-container">
    <a class="nav-item" href="/?page=Home">Home</a>
    <a class="nav-item" href="/?page=Inventory">Inventory</a>
    <a class="nav-item" href="/?page=Search">Search</a>
    <a class="nav-item" href="/?page=Profile">Profile</a>
</div>""", unsafe_allow_html=True)

current_page = st.query_params.get("page", "Home")

if current_page == "Home":
    st.title("TraidLive")
    st.markdown("<h3 style='color: #8E8E93;'>Welcome! Scan up to eight cards with one photo to find current listings.</h3>", unsafe_allow_html=True)
    
    source = st.radio("Scanner Source", ["Camera", "Gallery"], horizontal=True)
    img_input = st.camera_input("Scanner Active") if source == "Camera" else st.file_uploader("Drop Image")

    if img_input:
        with st.spinner("Processing..."):
            img_input.seek(0)
            asset_crops = detect_cards(img_input)
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
                        for label, grade in [("PSA 10", "PSA 10"), ("PSA 9", "PSA 9"), ("PSA 8", "PSA 8"), ("PSA 7", "PSA 7"), ("PSA 0-6", "PSA (1, 2, 3, 4, 5, 6)"), ("RAW", "Ungraded")]:
                            val, points = fetch_market_valuation(name, grade)
                            if not points:
                                html_rows += f"<tr style='border-bottom: 1px solid #1E2124;'><td style='padding: 10px 5px;'><strong>{label}</strong></td><td style='padding: 10px 5px; text-align: right; color:#FF453A; font-size: 12px;'>No listings</td></tr>"
                            else:
                                list_html = "".join([f"<div class='sold-row'><span class='sold-title'>{p['title']}</span><span class='sold-price'>${p['price']:,.2f}</span><a class='sold-link' href='{p['url']}' target='_blank'>Link</a></div>" for p in points])
                                act_link = f"https://www.ebay.com/sch/i.html?_nkw={requests.utils.quote(name + ' ' + grade)}"
                                list_html += f"<div style='margin-top:10px;'><a href='{act_link}' target='_blank' style='color:#000; background:#FFF; text-decoration:none; font-weight:700; padding:8px; border-radius:5px; display:block; text-align:center;'>🔍 Active Listings</a></div>"
                                html_rows += f"<tr style='border-bottom: 1px solid #1E2124;'><td style='padding: 10px 5px; vertical-align: top;'><strong>{label}</strong></td><td style='padding: 10px 5px; text-align: right;'><details><summary style='color:#00C805; font-weight:700;'>${val:,.2f} ▼</summary><div style='background:#151516; padding:10px; border-radius:8px; border:1px solid #3A3A3C; text-align:left; margin-top:10px;'>{list_html}</div></details></td></tr>"
                        
                        st.markdown(f"<div style='background:#1E2124; border-radius:12px; padding:10px; border:1px solid #30363D; margin-top:10px;'><table style='width:100%; border-collapse:collapse;'><thead><tr style='color:#8E8E93; border-bottom:1px solid #3A3A3C;'><th style='text-align:left; font-size:10px;'>GRADE</th><th style='text-align:right; font-size:10px;'>AVG PRICE</th></tr></thead><tbody>{html_rows}</tbody></table></div>", unsafe_allow_html=True)

# Other pages (Placeholders)
elif current_page == "Inventory": st.title("Vault")
elif current_page == "Search": st.title("Market Search")
elif current_page == "Profile": st.title("Profile")