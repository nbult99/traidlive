import streamlit as st
import requests
import base64
import pandas as pd

# --- 1. EBAY PRODUCTION LOGIC ---
def get_ebay_token():
    """Generates an OAuth token for eBay Production."""
    url = "https://api.ebay.com/identity/v1/oauth2/token"
    auth_str = f"{st.secrets['EBAY_APP_ID']}:{st.secrets['EBAY_CERT_ID']}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Authorization": f"Basic {encoded_auth}"}
    data = {"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}
    return requests.post(url, headers=headers, data=data).json().get("access_token")

def fetch_card_price(card_name):
    """Fetches real-time price averages from the Sports Trading Card category."""
    token = get_ebay_token()
    url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?q={card_name}&category_ids=212&limit=5"
    headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"}
    response = requests.get(url, headers=headers).json()
    items = response.get("itemSummaries", [])
    prices = [float(item['price']['value']) for item in items if 'price' in item]
    return sum(prices) / len(prices) if prices else None

# --- 2. SUPABASE INTEGRATION (DIAGNOSTIC MODE) ---
def save_to_supabase(card_name, price):
    """Saves data and returns a detailed status report."""
    url = f"{st.secrets['SUPABASE_URL']}/rest/v1/inventory"
    headers = {
        "apikey": st.secrets["SUPABASE_KEY"],
        "Authorization": f"Bearer {st.secrets['SUPABASE_KEY']}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"  # FORCES Supabase to show what it did
    }
    data = {"card_name": card_name, "market_price": price, "owner": "nbult99"}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code in [200, 201]:
            st.toast(f"Database confirmed: {response.json()[0]['card_name']} saved!")
            return True
        else:
            # Displays the exact reason for rejection (e.g., column mismatch, RLS)
            st.error(f"Supabase Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        st.error(f"Network Connection Failed: {str(e)}")
        return False

def get_inventory():
    """Fetches the current collection from Supabase."""
    url = f"{st.secrets['SUPABASE_URL']}/rest/v1/inventory?select=*"
    headers = {"apikey": st.secrets["SUPABASE_KEY"], "Authorization": f"Bearer {st.secrets['SUPABASE_KEY']}"}
    response = requests.get(url, headers=headers)
    return pd.DataFrame(response.json()) if response.status_code == 200 else pd.DataFrame()

# --- 3. MAIN UI ---
st.set_page_config(page_title="TraidLive", page_icon="📈")
st.title("📈 TraidLive Market Tracker")

card_input = st.text_input("Search Card Name:", value="LeBron James")

if st.button("PULL PRICE & SAVE"):
    with st.spinner("Pinging Marketplace..."):
        price = fetch_card_price(card_input)
        if price:
            st.metric("Current Market Avg", f"${price:,.2f}")
            if save_to_supabase(card_input, price):
                st.balloons()
                st.success("Successfully logged to Supabase!")
        else:
            st.warning("No sales found on eBay for this search.")

# --- 4. VIEW INVENTORY ---
st.divider()
st.subheader("📋 Your Digital Collection")
df = get_inventory()
if not df.empty:
    st.dataframe(df[['card_name', 'market_price', 'created_at']], use_container_width=True)
else:
    st.info("Inventory is empty. Use the search bar above to add your first card.")