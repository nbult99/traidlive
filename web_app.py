import streamlit as st
import cv2
import numpy as np
import requests
import base64
import pandas as pd
from PIL import Image

# --- 1. CORE API & STORAGE LOGIC ---

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

# --- 2. THE IMAGE RECOGNITION ENGINE ---

def detect_cards(image_file):
    """Yesterday's 1200px detection logic with confidence scoring."""
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Standardize to 1200px height for consistent detection
    ratio = 1200.0 / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * ratio), 1200))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crops = []
    # Pull the top 8 largest rectangular objects
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) == 4 and cv2.contourArea(cnt) > 5000:
            x, y, w, h = cv2.boundingRect(approx)
            crop = img[y:y+h, x:x+w]
            # Calculate a dummy confidence score based on rectangle perfection
            confidence = round((cv2.contourArea(cnt) / (w * h)) * 100, 1)
            crops.append({"image": crop, "confidence": confidence})
            
    return crops

# --- 3. UI & WORKFLOW ---

st.set_page_config(page_title="TraidLive Batch Scanner", page_icon="📷")
st.title("📷 TraidLive Batch Scanner")
st.write("Upload a photo with up to 8 cards for parallel processing.")

uploaded_image = st.file_uploader("Upload Collection Photo", type=['jpg', 'jpeg', 'png'])

if uploaded_image:
    with st.spinner("Analyzing cards..."):
        detected_items = detect_cards(uploaded_image)
        
    if detected_items:
        st.success(f"Found {len(detected_items)} potential cards.")
        
        # Grid Layout for detected cards
        cols = st.columns(2)
        for i, item in enumerate(detected_items):
            with cols[i % 2]:
                st.image(cv2.cvtColor(item['image'], cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"Detection Confidence: {item['confidence']}%")
                
                # Manual override if the OCR is still training
                card_id = st.text_input(f"Identify Card {i+1}:", value=f"LeBron James Card {i+1}")
                
                if st.button(f"Price & Save Card {i+1}"):
                    market_val = fetch_card_price(card_id)
                    if market_val:
                        if save_to_supabase(card_id, market_val):
                            st.success(f"Saved to Collection: ${market_val:,.2f}")
                        else:
                            st.error("Database connection issue.")
                    else:
                        st.warning("No market data found.")
    else:
        st.error("No cards detected. Try a photo with better contrast/lighting.")

# --- 4. DATA OVERVIEW ---
st.divider()
st.subheader("Your Inventory")
if st.button("Refresh Collection"):
    inventory_url = f"{st.secrets['SUPABASE_URL']}/rest/v1/inventory?select=*"
    headers = {"apikey": st.secrets["SUPABASE_KEY"], "Authorization": f"Bearer {st.secrets['SUPABASE_KEY']}"}
    data = requests.get(inventory_url, headers=headers).json()
    if data:
        st.table(pd.DataFrame(data)[['card_name', 'market_price', 'created_at']])