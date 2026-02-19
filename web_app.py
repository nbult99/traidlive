import streamlit as st
import requests
import base64

# --- 1. EBAY PRODUCTION LOGIC ---

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

def fetch_card_price(card_name):
    """Fetches real-time market data filtered to Sports Trading Cards (ID 212)."""
    token = get_ebay_token()
    if not token: 
        return "Error: Could not get eBay Token"
    
    # category_ids=212 targets 'Sports Trading Cards' only
    url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={card_name}&category_ids=212&limit=5"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"
    }
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        items = data.get("itemSummaries", [])
        
        if not items:
            return "No items found for this search."
            
        prices = [float(item['price']['value']) for item in items if 'price' in item]
        return sum(prices) / len(prices) if prices else "No pricing data available."
    except Exception as e:
        return f"API Error: {str(e)}"

# --- 2. MAIN USER INTERFACE ---

st.set_page_config(page_title="TraidLive Price Checker", page_icon="📈")

st.title("📈 TraidLive Market Checker")
st.write("Live Connection: **eBay Production Marketplace**")

# --- 3. THE LEBRON JAMES TRIAL ---
st.divider()
st.subheader("🧪 Live Trial")
card_to_search = st.text_input("Enter card name to search (e.g., LeBron James Rookie):", value="LeBron James")

if st.button("PULL LIVE EBAY PRICE"):
    with st.spinner(f"Querying eBay for '{card_to_search}'..."):
        result = fetch_card_price(card_to_search)
        
        if isinstance(result, float):
            st.balloons()
            st.success(f"Successfully pulled data from eBay!")
            st.metric(label=f"Avg Price for {card_to_search}", value=f"${result:,.2f}")
        else:
            st.error(result)

# --- 4. SYSTEM STATUS ---
st.divider()
st.caption("Logged in as: nbult99 | Status: Production Active")