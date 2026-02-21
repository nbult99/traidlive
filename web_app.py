import streamlit as st
import cv2
import numpy as np
import requests
import base64
import pandas as pd
import re # Added for strict attribute matching
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

# --- 2. THE MARKET ANALYSIS MODULE (STRICT VERSION) ---

def fetch_market_valuation(card_id_data, grade_filter=""):
    """
    Executes targeted marketplace searches with a 4-point exact-match verification.
    card_id_data: A dictionary containing {'year', 'brand', 'player', 'card_num'}
    """
    token_url = "https://api.ebay.com/identity/v1/oauth2/token"
    auth_str = f"{EBAY_APP_ID}:{EBAY_CERT_ID}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    
    # Extract strict attributes
    year = str(card_id_data.get('year', '')).strip()
    brand = str(card_id_data.get('brand', '')).strip()
    player = str(card_id_data.get('player', '')).strip()
    num = str(card_id_data.get('card_num', '')).strip()

    try:
        token_resp = requests.post(
            token_url, 
            headers={"Authorization": f"Basic {encoded_auth}", "Content-Type": "application/x-www-form-urlencoded"}, 
            data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}
        )
        token = token_resp.json().get("access_token")
        
        # 1. CONSTRUCT A BATTLE-HARDENED QUERY
        # We use quotes for the player and -reprint to kill fakes immediately
        search_query = f'{year} {brand} "{player}" #{num} {grade_filter} -reprint -rp -facsimile -digital -copy sold'
        query_encoded = requests.utils.quote(search_query)
        
        ebay_url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={query_encoded}&category_ids=212&limit=30"
        headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"}
        
        resp = requests.get(ebay_url, headers=headers)
        data = resp.json()
        items = data.get("itemSummaries", [])
        
        if not items: return 0.0, []
        
        # 2. THE FOUR-POINT INSPECTION LOOP
        points = []
        for item in items:
            title = item.get('title', '').upper()
            
            # Strict verification criteria
            has_year = year in title
            has_brand = any(b.upper() in title for b in brand.split())
            has_player = any(p.upper() in title for p in player.split())
            
            # Card Number match (checking for #236, 236, or No. 236 with word boundaries)
            has_num = re.search(rf"\b{num}\b", title) is not None
            
            # If it passes the 4-point check, keep it
            if has_year and has_brand and has_player and has_num:
                if 'price' in item:
                    points.append({
                        "title": item.get('title'),
                        "price": float(item['price']['value']),
                        "url": item.get('itemWebUrl', '#')
                    })
            
            if len(points) >= 10: break # Keep the top 10 most accurate
                
        if not points: return 0.0, []
        
        avg = sum(p['price'] for p in points) / len(points)
        return avg, points
    except:
        return 0.0, []

# --- 3. REFINED AI IDENTIFIER ---

def auto_label_crops(crops):
    if not hf_client: return []
    results = []
    for crop in crops:
        try:
            _, buf = cv2.imencode(".jpg", crop)
            b64 = base64.b64encode(buf.tobytes()).decode()
            
            # We now force the AI to output in a Pipe-Delimited format for easy parsing
            prompt = "Identify this sports card precisely. Return ONLY: Year | Brand | Player | Card #. Example: 2000 | Bowman | Tom Brady | 236"
            resp = hf_client.chat_completion(
                model=HF_MODEL, 
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], 
                max_tokens=60
            )
            
            raw_text = resp.choices[0].message.content.strip()
            parts = [p.strip() for p in raw_text.split('|')]
            
            if len(parts) >= 4:
                results.append({
                    "full_name": f"{parts[0]} {parts[1]} {parts[2]} #{parts[3]}",
                    "year": parts[0],
                    "brand": parts[1],
                    "player": parts[2],
                    "card_num": parts[3]
                })
            else:
                results.append({"full_name": "Identification Failed", "year": "", "brand": "", "player": "", "card_num": ""})
        except: 
            results.append({"full_name": "Vision Error", "year": "", "brand": "", "player": "", "card_num": ""})
    return results

# --- 4. TERMINAL UI (UPDATED TO HANDLE DICT RESULTS) ---

# [Design / Navigation code remains the same as your stable build...]

if page == "Home":
    st.title("TraidLive")
    # ... Scanner logic ...
    if img_input:
        with st.spinner("Analyzing Assets..."):
            img_input.seek(0)
            asset_crops = detect_cards(img_input)
            if st.button("AI BATCH IDENTIFY"):
                # Store the full dictionaries in session state
                st.session_state['scan_data'] = auto_label_crops(asset_crops)
        
        if 'scan_data' in st.session_state:
            st.divider()
            grid = st.columns(4)
            for i, card_dict in enumerate(st.session_state['scan_data']):
                with grid[i % 4]:
                    st.image(cv2.cvtColor(asset_crops[i], cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    # Allow user to edit the full identity string
                    display_name = st.text_input(f"Asset {i+1}", value=card_dict['full_name'], key=f"n_{i}")
                    
                    if st.button(f"CHECK MARKET {i+1}", key=f"p_{i}"):
                        # Re-parse if the user edited the text box manually
                        parts = [p.strip() for p in display_name.replace('#', '').split()]
                        # Fallback to dict if parse fails
                        search_data = card_dict if len(parts) < 4 else {
                            "year": parts[0], "brand": parts[1], "player": ' '.join(parts[2:-1]), "card_num": parts[-1]
                        }

                        with st.spinner("Executing 4-Point Inspection..."):
                            html_rows = ""
                            for label, query_filter in [("PSA 10", "PSA 10"), ("PSA 9", "PSA 9"), ("RAW", "Ungraded")]:
                                avg, points = fetch_market_valuation(search_data, query_filter)
                                # ... Table HTML generation logic remains same as stable build ...