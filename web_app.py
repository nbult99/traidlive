import streamlit as st
import requests
import base64

# --- 1. CORE API LOGIC ---

def get_ebay_token():
    """Generates a live OAuth token from eBay Production."""
    url = "https://api.ebay.com/identity/v1/oauth2/token"
    
    # Pulled from your Streamlit Secrets Vault
    app_id = st.secrets["EBAY_APP_ID"]
    cert_id = st.secrets["EBAY_CERT_ID"]
    
    auth_str = f"{app_id}:{cert_id}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {encoded_auth}"
    }
    data = {"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}
    
    try:
        response = requests.post(url, headers=headers, data=data)
        return response.json().get("access_token")
    except Exception:
        return None

def fetch_card_price(card_name, graded_only=False):
    """Fetches real-time market data filtered specifically to Category 212."""
    token = get_ebay_token()
    if not token: return None
    
    # Refining the query
    query = f"{card_name} PSA BGS Graded" if graded_only else card_name
    
    # category_ids=212 targets 'Sports Trading Cards' only
    url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={query}&category_ids=212&limit=5"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"
    }
    
    try:
        response = requests.get(url, headers=headers)
        items = response.json().get("itemSummaries", [])
        
        if not items:
            return None
            
        prices = [float(item['price']['value']) for item in items if 'price' in item]
        return sum(prices) / len(prices) if prices else None
    except Exception:
        return None

# --- 2. MAIN MOBILE UI ---

st.set_page_config(page_title="TraidLive", page_icon="🎴")

st.title("🎴 TraidLive")
st.write("Live Production Environment")

# Sidebar for user info
st.sidebar.title("Collector Profile")
st.sidebar.info("User: nbult99")

# --- 3. SYSTEM TRIAL SECTION (The Mickey Mantle Test) ---
st.divider()
st.subheader("🔧 System Diagnostic")
st.write("Run this trial to verify your live eBay Production connection.")

if st.button("RUN MICKEY MANTLE TRIAL"):
    with st.spinner("Pinging eBay Production Servers..."):
        trial_price = fetch_card_price("1952 Topps Mickey Mantle")
        
        if trial_price:
            st.balloons() # Success celebration!
            st.success("Handshake Successful! Connection is Live.")
            st.metric("1952 Mickey Mantle Avg", f"${trial_price:,.2f}")
        else:
            st.error("Diagnostic Failed.")
            st.info("Check if your EBAY_APP_ID in Secrets starts with 'NoahBult-traidliv-PRD'.")

# --- 4. SCANNER SECTION ---
st.divider()
st.subheader("📷 Card Scanner")
uploaded_file = st.camera_input("Scan Card")

if uploaded_file:
    # Placeholder for identified card name
    card_identity = "2023 Bowman Draft Tom Brady" 
    st.info(f"Analyzing: {card_identity}")
    
    graded = st.toggle("Filter for Graded (PSA/BGS)")
    
    if st.button("Get Real-Time Value"):
        with st.spinner("Searching Market..."):
            price = fetch_card_price(card_identity, graded_only=graded)
            if price:
                st.metric("Market Average", f"${price:.2f}")
            else:
                st.warning("No recent sales found for this specific card.")