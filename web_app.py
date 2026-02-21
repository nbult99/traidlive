import streamlit as st
import cv2
import numpy as np
import base64
import pandas as pd
import requests # Back to standard, reliable requests!
from huggingface_hub import InferenceClient
from supabase import create_client

# --- 1. INITIALIZATION & SECURE CONNECTIVITY ---
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
PSA_TOKEN = st.secrets.get("PSA_TOKEN", "")
EBAY_APP_ID = st.secrets.get("EBAY_APP_ID", "") # Re-activating your eBay API Key

@st.cache_resource
def init_connections():
    s_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    h_client = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else None
    return s_client, h_client

supabase, hf_client = init_connections()
HF_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# --- 2. THE iOS "OBSIDIAN" DESIGN SYSTEM ---
st.set_page_config(page_title="TraidLive | Market Auditor", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; font-family: -apple-system, sans-serif; }
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: #FFFFFF; }
    div[data-testid="stMetric"] { background: rgba(28, 28, 30, 0.8); border-radius: 18px; padding: 20px; border: 1px solid rgba(255, 255, 255, 0.1); }
    .stButton > button { background: linear-gradient(180deg, #0A84FF 0%, #007AFF 100%); color: white; border-radius: 12px; border: none; padding: 12px; font-weight: 600; width: 100%; }
    
    /* Audit Box Styling */
    .audit-box { background: #1C1C1E; border: 1px solid #3A3A3C; border-radius: 12px; padding: 15px; margin-top: 10px; font-size: 13px; }
    .audit-header { color: #8E8E93; font-size: 11px; text-transform: uppercase; margin-bottom: 10px; letter-spacing: 0.5px; }
    .sold-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #2C2C2E; }
    .sold-price { color: #34C759; font-weight: 700; }
    .sold-link { color: #0A84FF; text-decoration: none; font-weight: 500; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. OFFICIAL EBAY API (LIVE LISTINGS) ---
def fetch_market_valuation(card_name, grade_filter=""):
    app_id = EBAY_APP_ID
    if not app_id: return 0.0, []
    
    try:
        # Strip out the verification badges so it doesn't mess up the eBay search keywords
        clean_name = card_name.replace("✅ [PSA VERIFIED]", "").replace("❌ [Not PSA verified]", "").strip()
        search_query = f"{clean_name} {grade_filter}".strip()
        encoded_query = requests.utils.quote(search_query)
        
        # Using the official eBay Finding API for active items
        url = (
            f"https://svcs.ebay.com/services/search/FindingService/v1?"
            f"OPERATION-NAME=findItemsByKeywords&"
            f"SERVICE-VERSION=1.13.0&"
            f"SECURITY-APPNAME={app_id}&"
            f"RESPONSE-DATA-FORMAT=JSON&"
            f"REST-PAYLOAD=true&"
            f"keywords={encoded_query}&"
            f"categoryId=212" # Sports Mem, Cards & Fan Shop
        )
        
        resp = requests.get(url, timeout=10)
        data = resp.json()
        
        finding_response = data.get("findItemsByKeywordsResponse", [{}])[0]
        ack = finding_response.get("ack", [""])[0]
        
        if ack not in ["Success", "Warning"]:
            return 0.0, []
            
        search_result = finding_response.get("searchResult", [{}])[0]
        items = search_result.get("item", [])
        
        points = []
        for item in items[:10]: # Grab up to 10 live listings
            selling_status = item.get("sellingStatus", [{}])[0]
            price_str = selling_status.get("currentPrice", [{}])[0].get("__value__", "0")
            price = float(price_str)
            title = item.get("title", ["Unknown Card"])[0]
            item_url = item.get("viewItemURL", ["#"])[0]
            
            if price > 0:
                points.append({"title": title, "price": price, "url": item_url})
                
        if not points: return 0.0, []
        avg = sum(p['price'] for p in points) / len(points)
        return avg, points
        
    except Exception as e:
        return 0.0, []

# --- 4. CORE VISION & PSA VERIFICATION LOGIC ---
def verify_psa_cert(cert_number):
    if not PSA_TOKEN or not cert_number.isdigit():
        return False
        
    url = f"https://api.psacard.com/publicapi/cert/GetByCertNumber/{cert_number}"
    headers = {"Authorization": f"bearer {PSA_TOKEN}"}
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("PSACert", {}).get("CertNumber") is not None
        return False
    except:
        return False

def auto_label_crops(crops):
    if not hf_client: return ["" for _ in crops]
    labels = []
    for crop in crops:
        try:
            _, buf = cv2.imencode(".jpg", crop)
            b64 = base64.b64encode(buf.tobytes()).decode()
            
            prompt = "Identify this card. First line: Year, Brand, Player, Card #. Second line: If it is a graded PSA slab, output the 7 or 8 digit numeric certification number. If not, output 'NONE'."
            resp = hf_client.chat_completion(
                model=HF_MODEL, 
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], 
                max_tokens=60
            )
            
            result = resp.choices[0].message.content.strip().split('\n')
            card_name = result[0].strip()
            
            cert_num = ""
            if len(result) > 1:
                cert_num = ''.join(filter(str.isdigit, result[1]))
            
            if cert_num and len(cert_num) >= 7:
                if verify_psa_cert(cert_num):
                    labels.append(f"{card_name} ✅ [PSA VERIFIED]")
                else:
                    labels.append(f"{card_name} ❌ [Not PSA verified]")
            else:
                labels.append(f"{card_name} ❌ [Not PSA verified]")
                
        except: labels.append("Vision Error")
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
st.markdown("##### AI Portfolio Manager")

owner_id = st.sidebar.text_input("Customer ID", value="nbult99")

# LIVE PORTFOLIO CALCULATOR
try:
    vault_res = supabase.table("inventory").select("ungraded_price, psa_10_price").eq("owner", owner_id).execute()
    if vault_res.data:
        df_port = pd.DataFrame(vault_res.data)
        total_raw = df_port['ungraded_price'].sum()
        total_psa = df_port['psa_10_price'].sum()
        asset_count = len(df_port)
    else:
        total_raw, total_psa, asset_count = 0.0, 0.0, 0
except Exception:
    total_raw, total_psa, asset_count = 0.0, 0.0, 0

k1, k2, k3 = st.columns(3)
k1.metric("Raw Portfolio Value", f"${total_raw:,.2f}")
k2.metric("PSA 10 Potential", f"${total_psa:,.2f}")
k3.metric("Assets in Vault", asset_count)
st.divider()

source = st.radio("Asset Source", ["Gallery", "Camera"], horizontal=True)
uploaded_file = st.camera_input("Snap Card") if source == "Camera" else st.file_uploader("", type=['jpg','jpeg','png'])

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
                    db_name = name.replace("✅ [PSA VERIFIED]", "").replace("❌ [Not PSA verified]", "").strip()
                    p_psa, _ = fetch_market_valuation(db_name, "PSA 10")
                    p_raw, _ = fetch_market_valuation(db_name, "Ungraded")
                    supabase.table("inventory").insert({"card_name": db_name, "psa_10_price": p_psa, "ungraded_price": p_raw, "owner": owner_id}).execute()
                st.toast("Sync Complete.")
                st.rerun() 

        if 'suggestions' in st.session_state:
            st.divider()
            grid = st.columns(4)
            for i, crop in enumerate(asset_crops):
                with grid[i % 4]:
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.session_state['suggestions'][i] = st.text_input(f"ID {i+1}", value=st.session_state['suggestions'][i], key=f"inp_{i}")
                    
                    if st.button(f"Get Live Prices {i+1}", key=f"v_{i}"):
                        name = st.session_state['suggestions'][i]
                        for label, grade in [("PSA 10", "PSA 10"), ("RAW", "Ungraded")]:
                            avg, points = fetch_market_valuation(name, grade)
                            st.markdown(f"**{label} Avg: ${avg:,.2f}**")
                            
                            with st.container():
                                st.markdown(f'<div class="audit-box"><div class="audit-header">{label} ACTIVE LISTINGS</div>', unsafe_allow_html=True)
                                if not points:
                                    st.write("No active listings found.")
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

st.divider()
if st.button("Refresh Inventory Vault"):
    try:
        res = supabase.table("inventory").select("*").eq("owner", owner_id).order("created_at", desc=True).execute()
        if res.data:
            df = pd.DataFrame(res.data)
            st.dataframe(df[['card_name', 'ungraded_price', 'psa_10_price']], use_container_width=True)
    except Exception as e: st.error(f"Sync failed: {e}")