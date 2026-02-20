import streamlit as st
import cv2
import numpy as np
import requests
import base64
import pandas as pd
from huggingface_hub import InferenceClient
from supabase import create_client
from datetime import datetime, timedelta

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

# --- 2. THE iOS "OBSIDIAN" DESIGN SYSTEM ---
st.set_page_config(page_title="TraidLive | Sold Verification", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; font-family: -apple-system, sans-serif; }
    div[data-testid="stMetric"] { background: rgba(28, 28, 30, 0.8); border-radius: 18px; padding: 20px; border: 1px solid rgba(255, 255, 255, 0.1); }
    .stButton > button { background: linear-gradient(180deg, #0A84FF 0%, #007AFF 100%); color: white; border-radius: 14px; border: none; padding: 12px; font-weight: 600; width: 100%; }
    
    /* Audit Box Styling */
    .audit-box { 
        background: #1C1C1E; 
        border: 1px solid #3A3A3C; 
        border-radius: 12px; 
        padding: 15px; 
        margin-top: 10px; 
        font-size: 13px;
    }
    .audit-header { color: #8E8E93; font-size: 11px; text-transform: uppercase; margin-bottom: 10px; letter-spacing: 0.5px; }
    .sold-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #2C2C2E; }
    .sold-price { color: #34C759; font-weight: 700; }
    .sold-link { color: #0A84FF; text-decoration: none; font-weight: 500; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. THE "STRICT SOLD" ENGINE ---

def fetch_market_valuation(card_name, grade_filter=""):
    """
    FORCED COMPLETED SEARCH: Filters exclusively for items with a last_sold_date.
    """
    token_url = "https://api.ebay.com/identity/v1/oauth2/token"
    auth_str = f"{EBAY_APP_ID}:{EBAY_CERT_ID}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    
    try:
        # Auth Token
        token_resp = requests.post(token_url, 
                                 headers={"Authorization": f"Basic {encoded_auth}", "Content-Type": "application/x-www-form-urlencoded"}, 
                                 data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"})
        token = token_resp.json().get("access_token")
        
        # Look back 180 days to ensure we get data points for rare cards
        lookback = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%dT%H:%M:%S.000Z')
        search_query = f"{card_name} {grade_filter}"
        query_encoded = requests.utils.quote(search_query)
        
        # FILTER: last_sold_date:[RANGE] forces the API into the Completed index.
        ebay_url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={query_encoded}&filter=last_sold_date:[{lookback}..]&limit=5"
        
        headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"}
        resp = requests.get(ebay_url, headers=headers)
        data = resp.json()
        items = data.get("itemSummaries", [])
        
        if not items: return 0.0, []
            
        points = []
        for item in items:
            points.append({
                "title": item.get("title", "Unknown Card"),
                "price": float(item['price']['value']),
                "url": item.get("itemWebUrl", "#")
            })
            
        avg = sum(p['price'] for p in points) / len(points)
        return avg, points
        
    except: return 0.0, []

# --- 4. ORIGINAL LOGIC (RESTORED) ---

def auto_label_crops(crops):
    if not hf_client: return ["" for _ in crops]
    labels = []
    for crop in crops:
        try:
            _, buf = cv2.imencode(".jpg", crop)
            b64 = base64.b64encode(buf.tobytes()).decode()
            resp = hf_client.chat_completion(model=HF_MODEL, messages=[{"role": "user", "content": [{"type": "text", "text": "Identify this card. Return ONLY: Year, Brand, Player, Card #."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], max_tokens=50)
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

# --- 5. THE PROFESSIONAL INTERFACE ---

st.title("TraidLive")
st.markdown("##### Verified Sales Auditor")

owner_id = st.sidebar.text_input("Customer ID", value="nbult99")
source = st.radio("Asset Source", ["Gallery", "Camera"], horizontal=True)

uploaded_file = st.camera_input("Snap") if source == "Camera" else st.file_uploader("", type=['jpg','jpeg','png'])

if uploaded_file:
    with st.spinner("Analyzing Assets..."):
        uploaded_file.seek(0)
        asset_crops = detect_cards(uploaded_file)
    
    if asset_crops:
        col_ai, col_commit = st.columns(2)
        with col_ai:
            if st.button("AI Batch ID"):
                st.session_state['suggestions'] = auto_label_crops(asset_crops)
        with col_commit:
            if 'suggestions' in st.session_state and st.button("Commit All to Database"):
                for name in st.session_state['suggestions']:
                    p_psa, _ = fetch_market_valuation(name, "PSA 10")
                    p_raw, _ = fetch_market_valuation(name, "Ungraded")
                    supabase.table("inventory").insert({"card_name": name, "psa_10_price": p_psa, "ungraded_price": p_raw, "owner": owner_id}).execute()
                st.toast("Sync Complete.")

        if 'suggestions' in st.session_state:
            st.divider()
            grid = st.columns(4)
            for i, crop in enumerate(asset_crops):
                with grid[i % 4]:
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.session_state['suggestions'][i] = st.text_input(f"ID {i+1}", value=st.session_state['suggestions'][i], key=f"inp_{i}")
                    
                    if st.button(f"Verify Sales {i+1}", key=f"v_{i}"):
                        name = st.session_state['suggestions'][i]
                        
                        # Display PSA 10 and Raw Audits
                        for label, grade in [("PSA 10", "PSA 10"), ("RAW", "Ungraded")]:
                            avg, points = fetch_market_valuation(name, grade)
                            st.markdown(f"**{label} Avg: ${avg:,.2f}**")
                            
                            with st.container():
                                st.markdown(f'<div class="audit-box"><div class="audit-header">{label} RECENT SALES</div>', unsafe_allow_html=True)
                                if not points:
                                    st.write("No verified sales found in the last 180 days.")
                                else:
                                    for p in points:
                                        st.markdown(f'''
                                        <div class="sold-row">
                                            <span style="flex: 1; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; margin-right: 10px;">{p['title']}</span>
                                            <span class="sold-price">${p['price']:,.2f}</span>
                                            <a class="sold-link" href="{p['url']}" target="_blank" style="margin-left: 10px;">🔗</a>
                                        </div>
                                        ''', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)

# Ledger View
st.divider()
if st.button("Refresh Inventory Vault"):
    try:
        res = supabase.table("inventory").select("*").eq("owner", owner_id).order("created_at", desc=True).execute()
        if res.data:
            df = pd.DataFrame(res.data)
            st.dataframe(df[['card_name', 'ungraded_price', 'psa_10_price']], use_container_width=True)
    except Exception as e: st.error(f"Sync failed: {e}")