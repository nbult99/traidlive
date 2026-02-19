import streamlit as st
import cv2
import numpy as np
import requests
import base64
import pandas as pd
from PIL import Image

# --- 1. DATA PROCESSING LOGIC ---

def get_ebay_token():
    url = "https://api.ebay.com/identity/v1/oauth2/token"
    auth_str = f"{st.secrets['EBAY_APP_ID']}:{st.secrets['EBAY_CERT_ID']}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Authorization": f"Basic {encoded_auth}"}
    data = {"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}
    return requests.post(url, headers=headers, data=data).json().get("access_token")

def fetch_card_price(card_name):
    token = get_ebay_token()
    url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={card_name}&category_ids=212&limit=5"
    headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"}
    response = requests.get(url, headers=headers).json()
    items = response.get("itemSummaries", [])
    prices = [float(item['price']['value']) for item in items if 'price' in item]
    return sum(prices) / len(prices) if prices else None

def save_to_supabase(card_name, price):
    url = f"{st.secrets['SUPABASE_URL']}/rest/v1/inventory"
    headers = {
        "apikey": st.secrets["SUPABASE_KEY"],
        "Authorization": f"Bearer {st.secrets['SUPABASE_KEY']}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    data = {"card_name": card_name, "market_price": price, "owner": "nbult99"}
    response = requests.post(url, headers=headers, json=data)
    return response.status_code in [200, 201]

def detect_cards(image_file):
    """Detection logic utilizing 1200px standardization."""
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    ratio = 1200.0 / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * ratio), 1200))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 5000:
            x, y, w, h = cv2.boundingRect(approx)
            crop = img[y:y+h, x:x+w]
            confidence = round((cv2.contourArea(cnt) / (w * h)) * 100, 1)
            crops.append({"image": crop, "confidence": confidence})
    return crops

# --- 2. PROFESSIONAL USER INTERFACE ---

st.set_page_config(page_title="TraidLive | Professional Card Inventory", layout="wide")

# Custom CSS for a professional, minimalist look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        background-color: #004a99;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_value=True)

st.title("TraidLive Asset Management")
st.write("Production Market Analysis Environment")

# Sidebar for account overview
with st.sidebar:
    st.header("Account Overview")
    st.info("Authorized User: nbult99")
    st.write("System Status: Operational")

# --- 3. BATCH PROCESSING SECTION ---

st.subheader("Batch Asset Analysis")
uploaded_image = st.file_uploader("Upload portfolio image for detection (Maximum 8 assets)", type=['jpg', 'jpeg', 'png'])

if uploaded_image:
    with st.spinner("Analyzing image data..."):
        detected_items = detect_cards(uploaded_image)
        
    if detected_items:
        st.success(f"Analysis Complete: {len(detected_items)} assets identified.")
        
        # Professional Grid Layout
        cols = st.columns(4)
        for i, item in enumerate(detected_items):
            with cols[i % 4]:
                st.image(cv2.cvtColor(item['image'], cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"Confidence Rating: {item['confidence']}%")
                
                card_id = st.text_input(f"Identifier {i+1}", value=f"Asset {i+1}", key=f"input_{i}")
                
                if st.button(f"Commit Asset {i+1}", key=f"btn_{i}"):
                    market_val = fetch_card_price(card_id)
                    if market_val:
                        if save_to_supabase(card_id, market_val):
                            st.toast(f"Committed: {card_id} at ${market_val:,.2f}")
                        else:
                            st.error("Database commit error.")
                    else:
                        st.warning("Market valuation unavailable.")
    else:
        st.error("Identification failed. Please verify image lighting and contrast.")

# --- 4. INVENTORY RECORD ---

st.divider()
st.subheader("Current Inventory Records")

if st.button("Refresh Database Records"):
    inventory_url = f"{st.secrets['SUPABASE_URL']}/rest/v1/inventory?select=*"
    headers = {"apikey": st.secrets["SUPABASE_KEY"], "Authorization": f"Bearer {st.secrets['SUPABASE_KEY']}"}
    response = requests.get(inventory_url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            # Reordering columns for professional reporting
            df = df[['card_name', 'market_price', 'created_at']]
            df.columns = ['Asset Name', 'Market Value', 'Date Identified']
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No records currently exist in the database.")
    else:
        st.error("Authentication failed or connection timed out.")