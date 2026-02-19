import streamlit as st
import requests
import base64
import pandas as pd

# --- 1. EBAY PRODUCTION LOGIC ---
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

# --- 2. SUPABASE INTEGRATION ---
def save_to_supabase(card_name, price):
    url = f"{st.secrets['SUPABASE_URL']}/rest/v1/inventory"
    headers = {
        "apikey": st.secrets["SUPABASE_KEY"],
        "Authorization": f"Bearer {st.secrets['SUPABASE_KEY']}",
        "Content-Type": "application/json",
        "Prefer": "return=representation" 
    }
    data = {"card_name": card_name, "market_price": price, "owner": "nbult99"}
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code in [200, 201]:
        return True
    else:
        st.error(f"Save Failed: {response.status_code} - {response.text}")
        return False

def get_inventory():
    url = f"{st.secrets['SUPABASE_URL']}/rest/v1/inventory?select=*"
    headers = {"apikey": st.secrets["SUPABASE_KEY"], "Authorization": f"Bearer {st.secrets['SUPABASE_KEY']}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data) if data else pd.DataFrame()
    return pd.DataFrame()

# --- 3. MAIN UI ---
st.set_page_config(page_title="TraidLive", page_icon="📈")
st.title("📈 TraidLive Market Tracker")

card_input = st.text_input("Search and Save Card:", value="LeBron James")

if st.button("PULL & SAVE"):
    with st.spinner("Pinging Marketplace..."):
        price = fetch_card_price(card_input)
        if price:
            st.metric("Market Avg", f"${price:,.2f}")
            if save_to_supabase(card_input, price):
                st.balloons()
                st.success(f"Saved {card_input}!")
        else:
            st.error("Price not found on eBay.")

# --- 4. VIEW INVENTORY (Safety Version) ---
st.divider()
st.subheader("📋 Your Saved Collection")
inventory_df = get_inventory()

# CHECK: Does the data actually exist?
if not inventory_df.empty and 'card_name' in inventory_df.columns:
    st.dataframe(inventory_df[['card_name', 'market_price', 'created_at']], use_container_width=True)
else:
    st.info("Your collection is currently empty. The chart will appear once you save your first card.")