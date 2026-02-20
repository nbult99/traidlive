import streamlit as st
import cv2
import numpy as np
import requests
import base64
import pandas as pd
from huggingface_hub import InferenceClient
from supabase import create_client

# --- 1. INITIALIZATION ---
def get_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

supabase = get_supabase()

HF_TOKEN = st.secrets.get("HF_TOKEN", "")
# Use a highly reliable vision model for card identification
HF_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
hf_client = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else None

def auto_label_crops(crops):
    """Refined prompt to ensure eBay-compatible search terms."""
    if not hf_client:
        return ["" for _ in crops]
    labels = []
    for crop in crops:
        try:
            _, buf = cv2.imencode(".jpg", crop)
            b64 = base64.b64encode(buf.tobytes()).decode()
            # Stricter prompt for "Clean" search terms
            prompt = "Identify this trading card. Return ONLY: Year, Brand, Player Name, Card Number. No prose."
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

# --- 2. DATA HANDLERS ---

def fetch_card_price(card_name):
    """Isolates eBay connection logic for clean price retrieval."""
    url = "https://api.ebay.com/identity/v1/oauth2/token"
    auth_str = f"{st.secrets['EBAY_APP_ID']}:{st.secrets['EBAY_CERT_ID']}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    token_resp = requests.post(url, 
                               headers={"Authorization": f"Basic {encoded_auth}"}, 
                               data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"})
    token = token_resp.json().get("access_token")
    
    if not token: return None

    # Search Category 212 (Sports Cards)
    ebay_url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={card_name}&category_ids=212&limit=5"
    headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"}
    try:
        data = requests.get(ebay_url, headers=headers).json()
        items = data.get("itemSummaries", [])
        prices = [float(item['price']['value']) for item in items if 'price' in item]
        return sum(prices) / len(prices) if prices else None
    except:
        return None

# --- 3. UI DASHBOARD ---
st.set_page_config(page_title="TraidLive Asset Management", layout="wide")

st.title("TraidLive Asset Management")
owner_id = st.sidebar.text_input("Customer/Owner ID", value="nbult99")

uploaded_image = st.file_uploader("Upload Portfolio Image", type=['jpg', 'jpeg', 'png'])

if uploaded_image:
    # Card detection (keeping your existing OpenCV logic)
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    # ... (Insert your existing contour detection logic here) ...
    assets = [] # Placeholder for detected crops
    
    # Run Vision AI
    if st.button("Run AI Identification"):
        with st.spinner("Analyzing assets via Hugging Face..."):
            suggestions = auto_label_crops(assets)
            st.session_state['suggestions'] = suggestions

    if 'suggestions' in st.session_state:
        cols = st.columns(4)
        for i, crop in enumerate(assets):
            with cols[i % 4]:
                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                # Static key to prevent reset
                final_name = st.text_input(f"Label {i+1}", value=st.session_state['suggestions'][i], key=f"fixed_label_{i}")
                
                if st.button(f"Commit Asset {i+1}", key=f"commit_{i}"):
                    val = fetch_card_price(final_name)
                    if val:
                        # Direct Supabase insert
                        supabase.table("inventory").insert({"card_name": final_name, "market_price": val, "owner": owner_id}).execute()
                        st.success(f"Committed at ${val:,.2f}")