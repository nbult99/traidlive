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

# --- 2. CORE LOGIC MODULES ---

def fetch_market_valuation(card_name, grade_filter=""):
    """Targeted marketplace searches for SOLD items via eBay Production API."""
    token_url = "https://api.ebay.com/identity/v1/oauth2/token"
    auth_str = f"{EBAY_APP_ID}:{EBAY_CERT_ID}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    
    try:
        token_resp = requests.post(
            token_url, 
            headers={"Authorization": f"Basic {encoded_auth}", "Content-Type": "application/x-www-form-urlencoded"}, 
            data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}
        )
        token = token_resp.json().get("access_token")
        
        # Searching specifically for SOLD listings with the grade filter
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
    except:
        return 0.0

def auto_label_crops(crops):
    """Vision AI Identification for Batch Assets."""
    if not hf_client: return ["" for _ in crops]
    labels = []
    for crop in crops:
        try:
            _, buf = cv2.imencode(".jpg", crop)
            b64 = base64.b64encode(buf.tobytes()).decode()
            prompt = "Identify this card. Return ONLY: Year, Brand, Player, Card #. No prose."
            resp = hf_client.chat_completion(
                model=HF_MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}],
                max_tokens=50,
            )
            labels.append(resp.choices[0].message.content.strip())
        except:
            labels.append("")
    return labels

def detect_cards(image_file):
    """Computer Vision detection standardized at 1200px."""
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 1200px resize optimizes contour detection for standard card ratios
    ratio = 1200.0 / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * ratio), 1200))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crops = []
    # Identify top 8 rectangular regions based on surface area
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        # Ensure contour is a 4-sided polygon with sufficient area
        if len(approx) == 4 and cv2.contourArea(cnt) > 8000:
            x, y, w, h = cv2.boundingRect(approx)
            crops.append(img[y:y+h, x:x+w])
    return crops

# --- 3. PROFESSIONAL USER INTERFACE ---

st.set_page_config(page_title="TraidLive Asset Management", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    .stButton>button { width: 100%; background-color: #1a1a1a; color: white; border: none; padding: 10px; border-radius: 4px; }
    .stButton>button:hover { background-color: #333333; color: white; border: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("TraidLive Asset Management")
owner_id = st.sidebar.text_input("Customer ID", value="nbult99")

uploaded_file = st.file_uploader("Upload portfolio image for batch analysis (Max 8 assets)", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with st.spinner("Analyzing image contours..."):
        uploaded_file.seek(0)
        asset_crops = detect_cards(uploaded_file)
    
    if asset_crops:
        st.info(f"Analysis complete: {len(asset_crops)} assets identified.")
        
        if st.button("Run AI Batch Identification"):
            with st.spinner("Querying Vision Model for all assets..."):
                st.session_state['suggestions'] = auto_label_crops(asset_crops)

        if 'suggestions' in st.session_state:
            # Displays up to 8 cards in a clean professional grid
            cols = st.columns(4)
            for i, crop in enumerate(asset_crops):
                with cols[i % 4]:
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                    label = st.text_input(f"Identifier {i+1}", value=st.session_state['suggestions'][i], key=f"inp_{i}")
                    
                    if st.button(f"Commit Asset {i+1}", key=f"btn_{i}"):
                        with st.spinner(f"Valuing {label}..."):
                            psa_val = fetch_market_valuation(label, "PSA 10")
                            raw_val = fetch_market_valuation(label, "Ungraded")
                        
                        try:
                            supabase.table("inventory").insert({
                                "card_name": label, 
                                "psa_10_price": psa_val,
                                "ungraded_price": raw_val,
                                "owner": owner_id
                            }).execute()
                            st.toast(f"Synchronized: {label}")
                            st.write(f"PSA 10: ${psa_val:,.2f} | Raw: ${raw_val:,.2f}")
                        except Exception as e:
                            st.error(f"Sync Error: {str(e)}")
    else:
        st.warning("No card contours detected. Check contrast and lighting.")

# --- 4. INVENTORY LEDGER ---
st.divider()
st.subheader("Inventory Ledger")
if st.button("Refresh Database Records"):
    response = supabase.table("inventory").select("*").eq("owner", owner_id).order("created_at", desc=True).execute()
    if response.data:
        df = pd.DataFrame(response.data)
        cols_to_show = ['card_name', 'ungraded_price', 'psa_10_price', 'created_at']
        available_cols = [c for c in cols_to_show if c in df.columns]
        df_display = df[available_cols]
        df_display.columns = ['Asset Name', 'Ungraded Val', 'PSA 10 Val', 'Sync Date']
        st.dataframe(df_display, use_container_width=True)
    else:
        st.info("No records found for this Customer ID.")