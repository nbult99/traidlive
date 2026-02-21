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

# --- 2. MARKET ANALYSIS MODULES ---
def fetch_market_valuation(card_name, grade_filter=""):
    """
    Executes targeted marketplace searches.
    grade_filter: "PSA 10" or "Ungraded"
    """
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
    if not hf_client: return ["" for _ in crops]
    labels = []
    for crop in crops:
        try:
            _, buf = cv2.imencode(".jpg", crop)
            b64 = base64.b64encode(buf.tobytes()).decode()
            prompt = "Identify this card. Return ONLY: Year, Brand, Player, Card #. No prose."
            resp = hf_client.chat_completion(model=HF_MODEL, messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], max_tokens=50)
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

# --- 3. THE DARK MODE INTERFACE ---

st.set_page_config(page_title="TraidLive | Digital Assets", layout="wide")

# iOS Dark Mode CSS Injection
st.markdown("""
    <style>
    /* Dark background */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    /* Rounded Card Style for text inputs */
    div.stTextInput > div > div > input {
        background-color: #1C1C1E;
        color: white;
        border: 1px solid #3A3A3C;
        border-radius: 10px;
    }
    /* iOS Style Buttons */
    .stButton > button {
        background-color: #0A84FF; /* iOS System Blue */
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #409CFF;
        color: white;
        transform: scale(1.02);
    }
    /* Price preview text styling */
    .price-preview {
        background-color: #1C1C1E;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #3A3A3C;
        margin-top: 10px;
        font-size: 14px;
    }
    /* Dataframe styling */
    .stDataFrame {
        background-color: #1C1C1E;
        border-radius: 15px;
    }
    /* Header fonts */
    h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1C1C1E;
        border-right: 1px solid #3A3A3C;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("TraidLive")
st.write("Market Intelligence Dashboard")

owner_id = st.sidebar.text_input("Customer ID", value="nbult99")

uploaded_file = st.file_uploader("Upload Collection Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with st.spinner("Analyzing image..."):
        uploaded_file.seek(0)
        asset_crops = detect_cards(uploaded_file)
    
    if asset_crops:
        st.info(f"{len(asset_crops)} assets detected in frame.")
        
        col_ai, col_commit_all = st.columns(2)
        
        with col_ai:
            if st.button("AI Batch Identification"):
                with st.spinner("Processing..."):
                    st.session_state['suggestions'] = auto_label_crops(asset_crops)

        with col_commit_all:
            if 'suggestions' in st.session_state:
                if st.button("Commit All to Inventory"):
                    with st.spinner("Synchronizing..."):
                        for i, name in enumerate(st.session_state['suggestions']):
                            psa_val = fetch_market_valuation(name, "PSA 10")
                            raw_val = fetch_market_valuation(name, "Ungraded")
                            supabase.table("inventory").insert({"card_name": name, "psa_10_price": psa_val, "ungraded_price": raw_val, "owner": owner_id}).execute()
                        st.success("Batch successfully committed.")

        if 'suggestions' in st.session_state:
            st.divider()
            cols = st.columns(4)
            for i, crop in enumerate(asset_crops):
                with cols[i % 4]:
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.session_state['suggestions'][i] = st.text_input(f"Asset {i+1}", value=st.session_state['suggestions'][i], key=f"inp_{i}")
                    
                    # NEW: Individual Price Check Button
                    if st.button(f"Check Price {i+1}", key=f"chk_{i}"):
                        name = st.session_state['suggestions'][i]
                        with st.spinner("Fetching..."):
                            psa_val = fetch_market_valuation(name, "PSA 10")
                            raw_val = fetch_market_valuation(name, "Ungraded")
                            
                            st.markdown(f"""
                            <div class="price-preview">
                                <span style='color:#34C759; font-weight:bold;'>PSA 10:</span> ${psa_val:,.2f}<br>
                                <span style='color:#0A84FF; font-weight:bold;'>Raw:</span> ${raw_val:,.2f}
                            </div>
                            """, unsafe_allow_html=True)
    else:
        st.warning("No card contours detected.")

# --- 4. INVENTORY RECORD ---
st.divider()
st.subheader("Inventory Ledger")
if st.button("Refresh Database"):
    try:
        response = supabase.table("inventory").select("*").eq("owner", owner_id).order("created_at", desc=True).execute()
        if response.data:
            df = pd.DataFrame(response.data)
            cols_to_show = ['card_name', 'ungraded_price', 'psa_10_price', 'created_at']
            available_cols = [c for c in cols_to_show if c in df.columns]
            df_display = df[available_cols]
            df_display.columns = ['Asset Name', 'Ungraded Val', 'PSA 10 Val', 'Sync Date']
            st.dataframe(df_display, use_container_width=True)
        else:
            st.info("No records found.")
    except Exception as e:
        st.error(f"Query failed: {str(e)}")