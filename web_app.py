import streamlit as st
import cv2
import numpy as np
import requests
import base64
import pandas as pd
from PIL import Image

# --- 1. DATA INFRASTRUCTURE ---

def get_ebay_token():
    url = "https://api.ebay.com/identity/v1/oauth2/token"
    auth_str = f"{st.secrets['EBAY_APP_ID']}:{st.secrets['EBAY_CERT_ID']}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Authorization": f"Basic {encoded_auth}"}
    data = {"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}
    return requests.post(url, headers=headers, data=data).json().get("access_token")

def fetch_card_price(card_name):
    token = get_ebay_token()
    # Sports Trading Cards (Category 212)
    url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={card_name}&category_ids=212&limit=5"
    headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"}
    try:
        response = requests.get(url, headers=headers).json()
        items = response.get("itemSummaries", [])
        prices = [float(item['price']['value']) for item in items if 'price' in item]
        return sum(prices) / len(prices) if prices else None
    except:
        return None

def save_to_inventory(card_name, price):
    """The permanent handshake between the scan and your database."""
    url = f"{st.secrets['SUPABASE_URL']}/rest/v1/inventory"
    headers = {
        "apikey": st.secrets["SUPABASE_KEY"],
        "Authorization": f"Bearer {st.secrets['SUPABASE_KEY']}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    # Matches your SQL schema: card_name, market_price, owner
    data = {
        "card_name": card_name, 
        "market_price": price, 
        "owner": "nbult99"
    }
    response = requests.post(url, headers=headers, json=data)
    return response.status_code in [200, 201]

def detect_cards(image_file):
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
            crops.append(img[y:y+h, x:x+w])
    return crops

# --- 2. THE DASHBOARD ---

st.set_page_config(page_title="TraidLive Asset Management", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .stButton>button { width: 100%; background-color: #004a99; color: white; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

st.title("TraidLive Asset Management")
st.write("Secure Database Synchronization Active")

# --- 3. THE SCANNER WORKFLOW ---

st.subheader("Batch Asset Identification")
uploaded_image = st.file_uploader("Upload Image (Max 8 Assets)", type=['jpg', 'jpeg', 'png'])

if uploaded_image:
    assets = detect_cards(uploaded_image)
    if assets:
        st.success(f"Identification successful: {len(assets)} items detected.")
        cols = st.columns(4)
        for i, crop in enumerate(assets):
            with cols[i % 4]:
                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                asset_label = st.text_input(f"Asset Label {i+1}", value=f"Asset {i+1}", key=f"label_{i}")
                
                if st.button(f"Commit Asset {i+1}", key=f"commit_{i}"):
                    val = fetch_card_price(asset_label)
                    if val:
                        if save_to_inventory(asset_label, val):
                            st.toast(f"Synchronized: {asset_label} at ${val:,.2f}")
                        else:
                            st.error("Synchronization failed: Database link error.")
                    else:
                        st.warning("Valuation data unavailable for this identifier.")

# --- 4. THE INVENTORY VIEW ---

st.divider()
st.subheader("Inventory Ledger")

if st.button("Refresh Global Records"):
    inv_url = f"{st.secrets['SUPABASE_URL']}/rest/v1/inventory?select=*"
    headers = {"apikey": st.secrets["SUPABASE_KEY"], "Authorization": f"Bearer {st.secrets['SUPABASE_KEY']}"}
    response = requests.get(inv_url, headers=headers)
    if response.status_code == 200:
        df = pd.DataFrame(response.json())
        if not df.empty:
            df = df[['card_name', 'market_price', 'created_at']]
            df.columns = ['Asset Name', 'Market Value', 'Sync Date']
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No assets found in current inventory.")