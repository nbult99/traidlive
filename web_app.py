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
        "Content-Type": "application/json"
    }
    # Matches the columns you just created in the SQL Editor
    data = {"card_name": card_name, "market_price": price, "owner": "nbult99"}
    response = requests.post(url, headers=headers, json=data)
    return response.status_code in [200, 201]

def get_inventory():
    url = f"{st.secrets['SUPABASE_URL']}/rest/v1/inventory?select=*"
    headers = {"apikey": st.secrets["SUPABASE_KEY"], "Authorization": f"Bearer {st.secrets['SUPABASE_KEY']}"}
    response = requests.get(url, headers=headers)
    return pd.DataFrame(response.json()) if response.status_code == 200 else pd.DataFrame()

# --- 3. MAIN UI ---
st.title("📈 TraidLive Market Tracker")

card_input = st.text_input("Search and Save Card:", value="LeBron James Rookie")

if st.button("PULL PRICE & SAVE TO INVENTORY"):
    with st.spinner("Talking to eBay and Supabase..."):
        price = fetch_card_price(card_input)
        if price:
            if save_to_supabase(card_input, price):
                st.balloons()
                st.success(f"Success! Saved {card_input} at ${price:,.2f}")
            else:
                st.error("Price found, but Supabase rejected the save. Check RLS settings.")
        else:
            st.error("Could not find price on eBay.")

# --- 4. VIEW INVENTORY ---
st.divider()
st.subheader("📋 Your Saved Collection")
inventory_df = get_inventory()
if not inventory_df.empty:
    st.dataframe(inventory_df[['card_name', 'market_price', 'created_at']], use_container_width=True)
else:
    st.info("Your collection is currently empty. Save a card to see it here!")