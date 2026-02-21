import streamlit as st
import cv2
import numpy as np
import requests
import base64
import pandas as pd
import re
from huggingface_hub import InferenceClient
from supabase import create_client

# --- 1. INITIALIZATION ---
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
EBAY_APP_ID = st.secrets.get("EBAY_APP_ID", "")
EBAY_CERT_ID = st.secrets.get("EBAY_CERT_ID", "")
PSA_TOKEN = st.secrets.get("PSA_TOKEN", "")

@st.cache_resource
def init_connections():
    s_client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
    h_client = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else None
    return s_client, h_client

supabase, hf_client = init_connections()
HF_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# --- 2. ROBINHOOD DESIGN SYSTEM ---
st.set_page_config(page_title="TraidLive | Professional Card Terminal", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Global Styles */
    .stApp { background-color: #000000; color: #FFFFFF; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    
    /* Modern Right-Side Navigation Container */
    .nav-container {
        position: fixed;
        right: 30px;
        top: 50%;
        transform: translateY(-50%);
        z-index: 1000;
        display: flex;
        flex-direction: column;
        gap: 30px;
        background: rgba(30, 33, 36, 0.6);
        padding: 40px 20px;
        border-radius: 20px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .nav-item {
        color: rgba(255, 255, 255, 0.5);
        text-decoration: none;
        font-size: 15px;
        font-weight: 600;
        transition: 0.3s ease;
    }
    .nav-item:hover { color: #00C805; }

    /* Buttons & Inputs */
    .stButton > button {
        background-color: #00C805 !important;
        color: #000000 !important;
        border-radius: 24px !important;
        border: none !important;
        font-weight: 700 !important;
        transition: 0.2s ease;
        width: 100%;
        padding: 10px 0;
    }
    .stButton > button:hover { background-color: #00E606 !important; transform: translateY(-2px); }
    
    /* UI Cleanups */
    details > summary { list-style: none; outline: none; cursor: pointer; }
    .sold-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #1E2124; }
    .sold-price { color: #00C805; font-weight: 600; font-family: monospace; font-size: 1.1rem; }
    .welcome-text { color: #8E8E93; font-size: 1.2rem; margin-bottom: 30px; font-weight: 500; }
    
    /* Hide Default Streamlit Elements */
    [data-testid="stSidebar"] { display: none; }
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. THE UNIVERSAL SANITIZER ---
def sanitize_card_name(raw_input):
    clean = str(raw_input)
    if isinstance(raw_input, dict):
        clean = raw_input.get('full', str(raw_input))
    for artifact in ["{", "}", "'", ":", "full", "year", "brand", "player", "num", '"']:
        clean = clean.replace(artifact, "")
    return clean.strip()

# --- 4. RESILIENT MARKET ENGINE ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_valuation(clean_name, grade_filter=""):
    token_url = "https://api.ebay.com/identity/v1/oauth2/token"
    auth_str = f"{EBAY_APP_ID}:{EBAY_CERT_ID}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    
    # --- 1. SMART ATTRIBUTE EXTRACTION ---
    year_match = re.search(r'\b(19|20)\d{2}\b', clean_name)
    target_year = year_match.group(0) if year_match else ""
    
    num_match = re.search(r'\b\d{1,4}[A-Z]?\b$', clean_name.strip())
    target_num = num_match.group(0) if num_match else ""
    
    core_name = clean_name.replace(target_year, "").replace(target_num, "")
    for brand in ["PANINI", "TOPPS", "BOWMAN", "UPPER DECK", "PRIZM", "ABSOLUTE"]:
        core_name = re.sub(brand, '', core_name, flags=re.IGNORECASE)
    
    # FIX: Changed len(p) > 2 to len(p) > 1 to allow names like "Bo" or "Ty"
    player_parts = [p.upper() for p in core_name.strip().split() if len(p) > 1]

    try:
        token_resp = requests.post(token_url, headers={"Authorization": f"Basic {encoded_auth}"}, 
                                   data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}, timeout=5)
        token = token_resp.json().get("access_token")
        
        # --- 2. HIERARCHICAL QUERIES ---
        queries = [
            f"{clean_name} {grade_filter} -reprint -rp",
            f"{' '.join(player_parts)} {target_num} {target_year} {grade_filter} -reprint -rp"
        ]
        
        points = [] # Move points array outside to collect valid results across attempts
        
        for q in queries:
            if grade_filter == "Ungraded":
                # FIX: Shortened negative keyword list to prevent API choking
                q = q.replace("Ungraded", "") + " -PSA -BGS -SGC -CGC" 
            
            q = re.sub(r'\s+', ' ', q).strip()
            
            ebay_url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={requests.utils.quote(q)}&category_ids=212&limit=40"
            resp = requests.get(ebay_url, headers={"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"}, timeout=5)
            items = resp.json().get("itemSummaries", [])
            
            # FIX: Validate items immediately. 
            for item in items:
                title = item.get('title', '').upper()
                
                # A. Grade check (Mandatory)
                grade_ok = True
                if "PSA" in grade_filter:
                    all_psa = ["PSA 10", "PSA 9", "PSA 8", "PSA 7", "PSA 6"]
                    if any(g in title for g in all_psa if g != grade_filter.upper()):
                        grade_ok = False
                
                # B. Strict Verification
                has_year = target_year in title if target_year else True
                has_num = True
                if target_num:
                    has_num = re.search(rf"(?:^|\D){re.escape(target_num)}(?:\D|$)", title) is not None
                    
                has_player = all(p in title for p in player_parts) if player_parts else True

                # The final strict gate
                if grade_ok and has_year and has_num and has_player:
                    if 'price' in item:
                        points.append({
                            "title": item['title'], 
                            "price": float(item['price']['value']), 
                            "url": item.get('itemWebUrl', '#')
                        })
                
                if len(points) >= 10: break # Stop if we found 10 good comps
            
            # FIX: Only break out of the query loop if we actually successfully validated items!
            if points: 
                break
                
        if not points: return 0.0, []
        return (sum(p['price'] for p in points) / len(points)), points
        
    except Exception as e:
        print(f"Error fetching market data: {e}") 
        return 0.0, []
def auto_label_crops(crops):
    if not hf_client:
        return ["API Key Missing"] * len(crops)
        
    labels = []
    # UPDATED PROMPT: Force OCR and explicitly ban guessing.
    prompt = """OCR ONLY: Read the text on this card. Return ONLY: [Year] [Brand] [Player] [Card #]. DO NOT GUESS based on team or uniform. 
    If the card is in a PSA graded slab, also find the certification number (7 to 9 digits) on the label and append it exactly like this: | PSA: 123456789.
    Example: 2024 Panini Prizm Drake Maye 301 | PSA: 84729103"""
    
    for crop in crops:
        try:
            _, buf = cv2.imencode(".jpg", crop)
            b64 = base64.b64encode(buf.tobytes()).decode()
            
            resp = hf_client.chat_completion(
                model=HF_MODEL, 
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], 
                max_tokens=60,
                temperature=0.1 # Lower temperature makes the AI less "creative" and more analytical
            )
            labels.append(resp.choices[0].message.content.strip())
        except Exception as e: 
            print(f"HF Error: {e}")
            labels.append("ID Error")
    return labels
@st.cache_data(ttl=86400, show_spinner=False)
def generate_card_history(card_name):
    if not hf_client:
        return "API Key Missing. Cannot generate history."
    prompt = f"Write a short, engaging 2-3 sentence paragraph about the history and significance of this sports card: {card_name}. Do not include formatting, just the text."
    try:
        resp = hf_client.chat_completion(
            model=HF_MODEL, 
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}], 
            max_tokens=150,
            temperature=0.6
        )
        return resp.choices[0].message.content.strip()
    except Exception as e: 
        return f"Could not generate history: {e}"
@st.cache_data(ttl=86400, show_spinner=False)
def verify_psa_cert(cert_number):
    if not PSA_TOKEN:
        return False, "PSA API Token missing from secrets"
        
    url = f"https://api.psacard.com/publicapi/cert/GetByCertNumber/{cert_number}"
    headers = {"authorization": f"bearer {PSA_TOKEN}"}
    
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("IsValidRequest"):
                return True, "Valid"
        return False, "Not Found"
    except Exception as e:
        return False, str(e)
# --- 6. DATABASE OPERATIONS ---
def save_card_to_vault(user_id, card_name, psa_cert, grade, price):
    try:
        data = {
            "user_id": user_id,
            "card_name": card_name,
            "psa_cert": psa_cert,
            "grade": grade,
            "last_value": price
        }
        supabase.table("card_vault").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Error saving card: {e}")
        return False

def get_user_vault(user_id):
    try:
        response = supabase.table("card_vault").select("*").execute()
        # Thanks to RLS, this automatically only returns THIS user's cards!
        return response.data
    except Exception as e:
        return []
def detect_cards(image_file):
    image_file.seek(0) # Ensure file pointer is at the start
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    ratio = 1200.0 / img.shape[0]
    img_work = cv2.resize(img, (int(img.shape[1] * ratio), 1200))
    gray = cv2.cvtColor(img_work, cv2.COLOR_BGR2GRAY)
    
    # Improved image processing to find edges better
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 30, 150)
    dilated = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crops = []
    # Filter by Area AND Aspect Ratio to ensure it's actually a card
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area > 20000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = max(w, h) / float(min(w, h))
            
            # Standard cards have an aspect ratio around 1.4. This allows for some perspective warping.
            if 1.2 <= aspect_ratio <= 1.8:
                pad = 10
                y1, y2 = max(0, y-pad), min(img_work.shape[0], y+h+pad)
                x1, x2 = max(0, x-pad), min(img_work.shape[1], x+w+pad)
                crops.append(img_work[y1:y2, x1:x2])
                
                if len(crops) == 8: # Stop cleanly at 8 cards
                    break
    return crops

def display_pricing_table(raw_name):
    clean_name = sanitize_card_name(raw_name)
    html_rows = ""
    with st.spinner(f"Pulling comps for {clean_name}..."):
        for label, gr_query in [("PSA 10", "PSA 10"), ("PSA 9", "PSA 9"), ("PSA 8", "PSA 8"), ("RAW", "Ungraded")]:
            val, points = fetch_market_valuation(clean_name, gr_query)
            if points:
                list_html = "".join([f"<div class='sold-row'><span class='sold-title'>{p['title']}</span><span class='sold-price'>${p['price']:,.2f}</span><a href='{p['url']}' target='_blank' style='color:#0A84FF; text-decoration:none;'>Link</a></div>" for p in points])
                active_search = f"{clean_name} {gr_query}"
                if gr_query == "Ungraded": active_search = f"{clean_name} -graded"
                active_link = f"https://www.ebay.com/sch/i.html?_nkw={requests.utils.quote(active_search)}"
                list_html += f"<div style='margin-top:10px;'><a href='{active_link}' target='_blank' style='color:#000; background:#FFF; text-decoration:none; font-weight:700; padding:8px; border-radius:5px; display:block; text-align:center;'>🔍 View Active Listings</a></div>"
                html_rows += f"<tr><td style='padding:10px 0;'><strong>{label}</strong></td><td style='text-align:right;'><details><summary style='color:#00C805; font-weight:bold;'>${val:,.2f} ▼</summary><div style='background:#151516; padding:10px; border-radius:8px; border:1px solid #3A3A3C; text-align:left; margin-top:10px;'>{list_html}</div></details></td></tr>"
        
    if html_rows:
        st.markdown(f"<div style='background:#1E2124; border-radius:12px; padding:15px; border:1px solid #30363D; margin-top:10px;'><table style='width:100%; border-collapse:collapse;'>{html_rows}</table></div>", unsafe_allow_html=True)
    else:
        st.warning(f"Market analysis failed for: {clean_name}. Ensure spelling is correct or refine the search.")

# --- 5. INTERFACE ---
st.markdown("""
<div class="nav-container">
    <a class="nav-item" href="/?page=Home" target="_self">Home</a>
    <a class="nav-item" href="/?page=Vault" target="_self">Vault</a>
    <a class="nav-item" href="/?page=Trending" target="_self">Trending</a>
    <a class="nav-item" href="/?page=Profile" target="_self">Profile</a>
    <a class="nav-item" href="/?page=About" target="_self">About Us</a>
</div>
""", unsafe_allow_html=True)

page = st.query_params.get("page", "Home")

# Restrict the main content width so it doesn't overlap with the right-side navigation
main_col, empty_right_pad = st.columns([0.85, 0.15])

with main_col:
    if page == "Home":
        st.title("TraidLive")
        
        # 1. Custom Welcome Greeting
        if st.session_state.get('logged_in') and 'user' in st.session_state:
            username = st.session_state['user'].email.split('@')[0]
            st.markdown(f'<p class="welcome-text">Welcome back, {username}! Upload a photo with up to 8 cards to search current listings.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="welcome-text">Welcome! Upload a photo with up to 8 cards to search current listings.</p>', unsafe_allow_html=True)
        
        search_tab, upload_tab = st.tabs(["🔍 Search with Text", "📸 Upload Photo"])
        
        with search_tab:
            st.markdown("<br>", unsafe_allow_html=True)
            search_q = st.text_input("Quick Lookup", placeholder="e.g. 2024 Panini Prizm Drake Maye 301")
            if search_q and st.button("GET MARKET DATA", key="text_search_btn"):
                display_pricing_table(search_q)

        with upload_tab:
            st.markdown("<br>", unsafe_allow_html=True)
            upload_option = st.radio("Choose Photo Source:", ["Use Camera", "Upload Document"], horizontal=True)
            
            img_input = None
            if upload_option == "Use Camera":
                # 2. Defer Camera Access Logic
                if not st.session_state.get('camera_active'):
                    if st.button("📷 Start Camera"):
                        st.session_state['camera_active'] = True
                        st.rerun()
                else:
                    img_input = st.camera_input("Scanner")
                    if st.button("❌ Close Camera"):
                        st.session_state['camera_active'] = False
                        st.rerun()
            else:
                img_input = st.file_uploader("Select Image", type=['jpg','jpeg','png'])
                st.session_state['camera_active'] = False # Reset if they switch tabs

            if img_input:
                asset_crops = detect_cards(img_input)
                if asset_crops:
                    st.success(f"Detected {len(asset_crops)} card(s).")
                    
                    if st.button("Identify Card/s and Check Price"):
                        with st.spinner("AI is analyzing text and fetching market data..."):
                            st.session_state['scan_results'] = auto_label_crops(asset_crops)
                            st.session_state['scan_crops'] = asset_crops
                            st.session_state['auto_price_check'] = True
                else:
                    st.warning("No cards detected. Try placing them on a contrasting background.")
                
                if 'scan_results' in st.session_state:
                    st.divider()
                    
                    # 3. Add All Cards to Vault Button
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button("💾 Save ALL to Vault", type="primary"):
                            if st.session_state.get('logged_in'):
                                saved_count = 0
                                with st.spinner("Saving collection to vault..."):
                                    for name in st.session_state['scan_results']:
                                        display_name = name
                                        psa_cert = "None"
                                        if "| PSA:" in name:
                                            parts = name.split("| PSA:")
                                            display_name = parts[0].strip()
                                            psa_cert = re.sub(r'\D', '', parts[1])
                                            
                                        # Quick price fetch for saving
                                        val, _ = fetch_market_valuation(display_name, "Ungraded")
                                        success = save_card_to_vault(st.session_state['user'].id, display_name, psa_cert, "RAW", val)
                                        if success: saved_count += 1
                                st.success(f"Saved {saved_count} cards to your Vault!")
                            else:
                                st.error("Please log in via the Profile tab to save cards.")

                    st.markdown("<br>", unsafe_allow_html=True)
                    grid = st.columns(4)
                    
                    for i, name in enumerate(st.session_state['scan_results']):
                        with grid[i % 4]:
                            st.image(cv2.cvtColor(st.session_state['scan_crops'][i], cv2.COLOR_BGR2RGB), use_container_width=True)
                            
                            display_name = name
                            psa_cert = None
                            if "| PSA:" in name:
                                parts = name.split("| PSA:")
                                display_name = parts[0].strip()
                                psa_cert = re.sub(r'\D', '', parts[1])
                            
                            current_name = st.text_input(f"Asset {i+1}", value=sanitize_card_name(display_name), key=f"sn_{i}")
                            
                            if psa_cert and len(psa_cert) >= 7:
                                is_valid, status = verify_psa_cert(psa_cert)
                                if is_valid:
                                    st.markdown(f"✅ **Verified:** [PSA {psa_cert}](https://www.psacard.com/cert/{psa_cert})", unsafe_allow_html=True)
                                elif status == "API_OFFLINE":
                                    st.markdown(f"🔗 [View on PSA Website](https://www.psacard.com/cert/{psa_cert})", unsafe_allow_html=True)
                                elif status == "Not Found":
                                    st.markdown(f"⚠️ **PSA Alert:** {psa_cert} (Not Found)", unsafe_allow_html=True)

                            if st.button(f"Update Price Check", key=f"sp_{i}"):
                                st.session_state['auto_price_check'] = True
                                
                            if st.session_state.get('auto_price_check'):
                                val, _ = fetch_market_valuation(current_name, "Ungraded")
                                display_pricing_table(current_name)
                                
                                if st.button("🤖 About this card", key=f"about_{i}"):
                                    with st.spinner("Consulting AI Lore..."):
                                        st.session_state[f"history_{i}"] = generate_card_history(current_name)
                                
                                if f"history_{i}" in st.session_state:
                                    st.info(st.session_state[f"history_{i}"])
                                
                                # 4. Individual Save Button
                                if st.button("💾 Save to Vault", key=f"save_{i}"):
                                    if st.session_state.get('logged_in'):
                                        success = save_card_to_vault(
                                            user_id=st.session_state['user'].id,
                                            card_name=current_name,
                                            psa_cert=psa_cert if psa_cert else "None",
                                            grade="RAW",
                                            price=val
                                        )
                                        if success: st.toast("✅ Asset secured in Vault!")
                                    else:
                                        st.warning("Please log in via the Profile tab to save cards.")

    # 5. The New Vault Page
    elif page == "Vault":
        st.title("My Vault")
        if not st.session_state.get('logged_in'):
            st.warning("You must be logged in to view and manage your Vault.")
            st.markdown('<a href="/?page=Profile" target="_self"><button style="background:#00C805; color:black; padding:10px 20px; border-radius:8px; border:none; font-weight:bold; cursor:pointer;">Go to Login</button></a>', unsafe_allow_html=True)
        else:
            user_cards = get_user_vault(st.session_state['user'].id)
            if user_cards:
                import pandas as pd
                df = pd.DataFrame(user_cards)
                df = df[['card_name', 'grade', 'psa_cert', 'last_value', 'created_at']]
                df.columns = ["Asset", "Grade", "PSA Cert", "Est. Value", "Date Added"]
                df["Est. Value"] = df["Est. Value"].apply(lambda x: f"${float(x):,.2f}" if pd.notnull(x) else "$0.00")
                df["Date Added"] = pd.to_datetime(df["Date Added"]).dt.strftime('%Y-%m-%d')
                
                st.dataframe(df, hide_index=True, use_container_width=True)
                total_val = sum(c['last_value'] for c in user_cards if c.get('last_value'))
                st.markdown(f"<h3 style='color:#00C805;'>Total Vault Value: ${total_val:,.2f}</h3>", unsafe_allow_html=True)
            else:
                st.info("Your vault is empty. Scan cards on the Home page to start building your collection!")

    elif page == "Trending":
        st.title("Trending Cards:")
        st.markdown("<p style='color: #8E8E93; margin-top: 20px;'>Market movement data populating soon...</p>", unsafe_allow_html=True)

    elif page == "Profile":
        st.title("Account Settings")
        
        if 'logged_in' not in st.session_state:
            st.session_state['logged_in'] = False

        if not st.session_state['logged_in']:
            login_tab, register_tab = st.tabs(["Login", "Create New Profile"])
            
            with login_tab:
                st.markdown("<br>", unsafe_allow_html=True)
                email = st.text_input("Email", key="log_user")
                pwd = st.text_input("Password", type="password", key="log_pass")
                if st.button("Login", key="btn_login"):
                    with st.spinner("Authenticating..."):
                        try:
                            res = supabase.auth.sign_in_with_password({"email": email, "password": pwd})
                            st.session_state['user'] = res.user
                            st.session_state['logged_in'] = True
                            st.rerun()
                        except Exception:
                            st.error("Invalid email or password.")
                    
            with register_tab:
                st.markdown("<br>", unsafe_allow_html=True)
                new_email = st.text_input("Email Address", key="reg_email")
                new_pwd = st.text_input("Create Password (Min 6 Characters)", type="password", key="reg_pass")
                if st.button("Create Profile", key="btn_reg"):
                    with st.spinner("Creating secure profile..."):
                        try:
                            res = supabase.auth.sign_up({"email": new_email, "password": new_pwd})
                            st.success("Profile Created! Check your email for a confirmation link (if enabled) or Login.")
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            st.success(f"Logged in securely as: {st.session_state['user'].email}")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Log Out"):
                supabase.auth.sign_out()
                st.session_state['logged_in'] = False
                st.session_state.pop('user', None)
                st.rerun()

    elif page == "About":
        st.title("About Us")
        st.markdown("""
        <div style="background: #1E2124; padding: 40px; border-radius: 16px; border: 1px solid #30363D; margin-top: 30px;">
            <p style="font-size: 1.2rem; line-height: 1.8; color: #E5E5EA; margin-bottom: 20px;">
                We are a company of card trading enthusiasts giving a database with AI card identification and price tracking to small businesses.
            </p>
            <p style="font-size: 1.1rem; line-height: 1.8; color: #8E8E93;">
                By combining state-of-the-art vision models with real-time market data, we provide a robust, automated workflow that takes the guesswork out of the hobby. Trade with confidence, price with precision, and build your vault.
            </p>
        </div>
        """, unsafe_allow_html=True)