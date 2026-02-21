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
    s_client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
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

# --- 3. THE UNIVERSAL SANITIZER ---
def sanitize_card_name(raw_input):
    clean = str(raw_input)
    if isinstance(raw_input, dict):
        clean = raw_input.get('full', str(raw_input))
    for artifact in ["{", "}", "'", ":", "full", "year", "brand", "player", "num", '"']:
        clean = clean.replace(artifact, "")
    return clean.strip()

# --- 4. RESILIENT MARKET ENGINE ---
def fetch_market_valuation(clean_name, grade_filter=""):
    token_url = "https://api.ebay.com/identity/v1/oauth2/token"
    auth_str = f"{EBAY_APP_ID}:{EBAY_CERT_ID}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    
    # --- 1. SMART ATTRIBUTE EXTRACTION ---
    year_match = re.search(r'\b(19|20)\d{2}\b', clean_name)
    target_year = year_match.group(0) if year_match else ""
    
    num_match = re.search(r'\b\d{1,4}[A-Z]?\b$', clean_name.strip())
    target_num = num_match.group(0) if num_match else ""
    
    core_name = clean_name.replace(target_year, "").replace(target_num, "")
    for brand in ["PANINI", "TOPPS", "BOWMAN", "UPPER DECK", "PRIZM", "ABSOLUTE"]:
        core_name = re.sub(brand, '', core_name, flags=re.IGNORECASE)
    
    player_parts = [p.upper() for p in core_name.strip().split() if len(p) > 2]

    try:
        token_resp = requests.post(token_url, headers={"Authorization": f"Basic {encoded_auth}"}, 
                                   data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}, timeout=5)
        token = token_resp.json().get("access_token")
        
        # --- 2. HIERARCHICAL QUERIES (Fixed) ---
        # Removed duplicate year and the word "sold"
        queries = [
            f"{clean_name} {grade_filter} -reprint -rp",
            f"{' '.join(player_parts)} {target_num} {target_year} {grade_filter} -reprint -rp"
        ]
        
        items = []
        for q in queries:
            if grade_filter == "Ungraded":
                q = q.replace("Ungraded", "") + " -PSA -BGS -SGC -CGC -graded"
            
            # Clean up double spaces that might occur from missing variables
            q = re.sub(r'\s+', ' ', q).strip()
            
            ebay_url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={requests.utils.quote(q)}&category_ids=212&limit=40"
            resp = requests.get(ebay_url, headers={"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"}, timeout=5)
            items = resp.json().get("itemSummaries", [])
            
            # If the strict query finds items, stop searching
            if items: break

        if not items: return 0.0, []
        
        points = []
        for item in items:
            title = item.get('title', '').upper()
            
            # --- 3. THE SMART VERIFICATION GATE (Fixed) ---
            
            # A. Grade check (Mandatory)
            grade_ok = True
            if "PSA" in grade_filter:
                all_psa = ["PSA 10", "PSA 9", "PSA 8", "PSA 7", "PSA 6"]
                if any(g in title for g in all_psa if g != grade_filter.upper()):
                    grade_ok = False
            
            # B. Strict Verification
            has_year = target_year in title if target_year else True
            
            # Use a more forgiving regex for the number to account for #301, No.301, etc.
            # It ensures the number isn't buried inside another longer number (like 3010)
            has_num = True
            if target_num:
                has_num = re.search(rf"(?:^|\D){target_num}(?:\D|$)", title) is not None
                
            has_player = all(p in title for p in player_parts) if player_parts else True

            # The final strict gate
            if grade_ok and has_year and has_num and has_player:
                if 'price' in item:
                    points.append({
                        "title": item['title'], 
                        "price": float(item['price']['value']), 
                        "url": item.get('itemWebUrl', '#')
                    })
            
            if len(points) >= 10: break
                
        if not points: return 0.0, []
        return (sum(p['price'] for p in points) / len(points)), points
        
    except Exception as e:
        print(f"Error fetching market data: {e}") 
        return 0.0, []
def auto_label_crops(crops):
    if not hf_client:
        return ["API Key Missing"] * len(crops)
        
    labels = []
    # UPDATED PROMPT: Force OCR and explicitly ban guessing.
    prompt = """
    Return ONLY a single line formatted strictly as: [Year] [Brand] [Player] [Card Number]
    Example: 2024 Panini Prizm Drake Maye 301
    Ensure the Card Number is at the very end.
    """
    
    for crop in crops:
        try:
            _, buf = cv2.imencode(".jpg", crop)
            b64 = base64.b64encode(buf.tobytes()).decode()
            
            resp = hf_client.chat_completion(
                model=HF_MODEL, 
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], 
                max_tokens=60,
                temperature=0.1 # Lower temperature makes the AI less "creative" and more analytical
            )
            labels.append(resp.choices[0].message.content.strip())
        except Exception as e: 
            print(f"HF Error: {e}")
            labels.append("ID Error")
    return labels

def detect_cards(image_file):
    image_file.seek(0) # Ensure file pointer is at the start
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    ratio = 1200.0 / img.shape[0]
    img_work = cv2.resize(img, (int(img.shape[1] * ratio), 1200))
    gray = cv2.cvtColor(img_work, cv2.COLOR_BGR2GRAY)
    
    # Improved image processing to find edges better
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 30, 150)
    dilated = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crops = []
    # Filter by Area AND Aspect Ratio to ensure it's actually a card
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area > 20000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = max(w, h) / float(min(w, h))
            
            # Standard cards have an aspect ratio around 1.4. This allows for some perspective warping.
            if 1.2 <= aspect_ratio <= 1.8:
                pad = 10
                y1, y2 = max(0, y-pad), min(img_work.shape[0], y+h+pad)
                x1, x2 = max(0, x-pad), min(img_work.shape[1], x+w+pad)
                crops.append(img_work[y1:y2, x1:x2])
                
                if len(crops) == 8: # Stop cleanly at 8 cards
                    break
    return crops

def display_pricing_table(raw_name):
    clean_name = sanitize_card_name(raw_name)
    html_rows = ""
    with st.spinner(f"Pulling comps for {clean_name}..."):
        for label, gr_query in [("PSA 10", "PSA 10"), ("PSA 9", "PSA 9"), ("PSA 8", "PSA 8"), ("RAW", "Ungraded")]:
            val, points = fetch_market_valuation(clean_name, gr_query)
            if points:
                list_html = "".join([f"<div class='sold-row'><span class='sold-title'>{p['title']}</span><span class='sold-price'>${p['price']:,.2f}</span><a href='{p['url']}' target='_blank' style='color:#0A84FF; text-decoration:none;'>Link</a></div>" for p in points])
                active_search = f"{clean_name} {gr_query}"
                if gr_query == "Ungraded": active_search = f"{clean_name} -graded"
                active_link = f"https://www.ebay.com/sch/i.html?_nkw={requests.utils.quote(active_search)}"
                list_html += f"<div style='margin-top:10px;'><a href='{active_link}' target='_blank' style='color:#000; background:#FFF; text-decoration:none; font-weight:700; padding:8px; border-radius:5px; display:block; text-align:center;'>🔍 View Active Listings</a></div>"
                html_rows += f"<tr><td style='padding:10px 0;'><strong>{label}</strong></td><td style='text-align:right;'><details><summary style='color:#00C805; font-weight:bold;'>${val:,.2f} ▼</summary><div style='background:#151516; padding:10px; border-radius:8px; border:1px solid #3A3A3C; text-align:left; margin-top:10px;'>{list_html}</div></details></td></tr>"
        
    if html_rows:
        st.markdown(f"<div style='background:#1E2124; border-radius:12px; padding:15px; border:1px solid #30363D; margin-top:10px;'><table style='width:100%; border-collapse:collapse;'>{html_rows}</table></div>", unsafe_allow_html=True)
    else:
        st.warning(f"Market analysis failed for: {clean_name}. Ensure spelling is correct or refine the search.")

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
    st.markdown("<h3 style='color: #8E8E93;'>Terminal v1.6 | Resilient Search Enabled</h3>", unsafe_allow_html=True)
    
    search_q = st.text_input("Quick Lookup", placeholder="e.g. 2000 Bowman Tom Brady 236")
    if search_q and st.button("GET MARKET DATA"):
        display_pricing_table(search_q)

    st.divider()

    input_type = st.radio("Input Method", ["Search with Text", "Upload Image"], horizontal=True)

    if input_type == "Upload Image":
        upload_option = st.radio("Source:", ["Browse Files", "Use Camera"], horizontal=True)
        img_input = st.camera_input("Scanner") if upload_option == "Use Camera" else st.file_uploader("Select Image", type=['jpg','jpeg','png'])

        if img_input:
            asset_crops = detect_cards(img_input)
            if asset_crops:
                st.success(f"Detected {len(asset_crops)} card(s).")
                if st.button("AI BATCH IDENTIFY"):
                    with st.spinner("AI is analyzing text on cards..."):
                        st.session_state['scan_results'] = auto_label_crops(asset_crops)
                        st.session_state['scan_crops'] = asset_crops
            else:
                st.warning("No cards detected. Try placing them on a contrasting background.")
            
            if 'scan_results' in st.session_state:
                grid = st.columns(4)
                for i, name in enumerate(st.session_state['scan_results']):
                    with grid[i % 4]:
                        st.image(cv2.cvtColor(st.session_state['scan_crops'][i], cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.session_state['scan_results'][i] = st.text_input(f"Asset {i+1}", value=sanitize_card_name(name), key=f"sn_{i}")
                        if st.button(f"Analyze {i+1}", key=f"sp_{i}"):
                            display_pricing_table(st.session_state['scan_results'][i])

elif page == "Inventory": 
    st.title("Vault")