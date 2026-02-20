import streamlit as st
import cv2
import numpy as np
import requests
import base64
import pandas as pd
from huggingface_hub import InferenceClient
from supabase import create_client

# --- 1. INITIALIZATION & SECURE CREDENTIALS ---

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

# --- 2. LOGIC MODULES ---

def auto_label_crops(crops):
    """Uses Vision AI to identify card details from image crops."""
    if not hf_client:
        st.error("Identification Error: Hugging Face Token is not configured.")
        return ["" for _ in crops]
    
    labels = []
    for crop in crops:
        try:
            _, buf = cv2.imencode(".jpg", crop)
            b64 = base64.b64encode(buf.tobytes()).decode()
            prompt = "Identify this trading card. Return ONLY: Year, Brand, Player Name, Card Number. Be concise."
            resp = hf_client.chat_completion(
                model=HF_MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}],
                max_tokens=50,
            )
            labels.append(resp.choices[0].message.content.strip())
        except Exception as e:
            st.warning(f"Vision analysis failed for an asset: {str(e)}")
            labels.append("")
    return labels

def fetch_ebay_price(card_name):
    """Diagnostic version of eBay price retrieval."""
    token_url = "https://api.ebay.com/identity/v1/oauth2/token"
    
    if not EBAY_APP_ID or not EBAY_CERT_ID:
        st.error("Credential Error: eBay credentials missing in secrets.")
        return None

    auth_str = f"{EBAY_APP_ID}:{EBAY_CERT_ID}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    
    try:
        # 1. Obtain Token
        token_resp = requests.post(
            token_url, 
            headers={"Authorization": f"Basic {encoded_auth}", "Content-Type": "application/x-www-form-urlencoded"}, 
            data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}
        )
        token = token_resp.json().get("access_token")
        
        if not token:
            st.error(f"Authentication Failed: {token_resp.json().get('error_description', 'Invalid Credentials')}")
            return None

        # 2. Execute Search
        query_encoded = requests.utils.quote(card_name)
        ebay_url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={query_encoded}&category_ids=212&limit=5"
        headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"}
        
        resp = requests.get(ebay_url, headers=headers)
        if resp.status_code != 200:
            st.error(f"Marketplace Search Failed: Status {resp.status_code}")
            return None
            
        data = resp.json()
        items = data.get("itemSummaries", [])
        
        if not items:
            st.warning(f"Market Gap: No current listings found for '{card_name}'.")
            return None
            
        prices = [float(item['price']['value']) for item in items if 'price' in item]
        return sum(prices) / len(prices)
    except Exception as e:
        st.error(f"System Exception during pricing: {str(e)}")
        return None

def detect_cards(image_file):
    """Standardized 1200px contour detection logic."""
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

uploaded_file = st.file_uploader("Upload portfolio image for batch analysis", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with st.spinner("Analyzing image contours..."):
        uploaded_file.seek(0)
        asset_crops = detect_cards(uploaded_file)
    
    if asset_crops:
        st.info(f"System identified {len(asset_crops)} assets.")
        
        if st.button("Run AI Identification"):
            with st.spinner("Querying Vision Model..."):
                st.session_state['suggestions'] = auto_label_crops(asset_crops)

        if 'suggestions' in st.session_state:
            cols = st.columns(4)
            for i, crop in enumerate(asset_crops):
                with cols[i % 4]:
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                    label = st.text_input(f"Asset {i+1}", value=st.session_state['suggestions'][i], key=f"inp_{i}")
                    
                    if st.button(f"Commit Asset {i+1}", key=f"btn_{i}"):
                        val = fetch_ebay_price(label)
                        if val:
                            try:
                                supabase.table("inventory").insert({
                                    "card_name": label, 
                                    "market_price": val, 
                                    "owner": owner_id
                                }).execute()
                                st.toast(f"Synchronized: {label} at ${val:,.2f}")
                            except Exception as e:
                                st.error(f"Sync Error: {str(e)}")
                        else:
                            st.warning("Valuation unavailable for this identifier.")
    else:
        st.warning("No clear card contours detected.")

# --- 4. INVENTORY RECORD ---
st.divider()
st.subheader("Inventory Ledger")
if st.button("Refresh Database"):
    try:
        response = supabase.table("inventory").select("*").eq("owner", owner_id).order("created_at", desc=True).execute()
        if response.data:
            df = pd.DataFrame(response.data)
            df = df[['card_name', 'market_price', 'created_at']]
            df.columns = ['Asset Name', 'Market Valuation', 'Timestamp']
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No records found for this ID.")
    except Exception as e:
        st.error(f"Query failed: {str(e)}")