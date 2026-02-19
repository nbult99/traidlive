import base64
import json
from PIL import Image
from io import BytesIO
from huggingface_hub import InferenceClient
from supabase import create_client, Client

# --- CONFIGURATION ---
SUPABASE_URL = "https://jtoobzfyeodbnrcojybk.supabase.co"
SUPABASE_KEY = "sb_publishable_gvxnNeZHgjuHqx-DuoFYhA_mzdQCqED"
HF_TOKEN = "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"

# Initialize tools
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = InferenceClient(api_key=HF_TOKEN)

# Using Qwen2.5-VL for high-precision multi-object detection
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

def identify_multiple_cards(image_path):
    print(f"🤖 AI is scanning {image_path} for cards...")
    
    # High resolution (1200px) is required to read text on multiple cards
    try:
        with Image.open(image_path) as img:
            img.thumbnail((1200, 1200))
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"❌ Image Error: {e}")
        return []

    prompt = """
    Identify every sports card in this image (up to 10). 
    For each card, provide: player, number, year, pack_set.
    Include a "confidence" score (0.0 to 1.0) for each identification.
    Return ONLY a valid JSON list of objects.
    Example: [{"player": "Drake Maye", "number": "103", "year": "2024", "pack_set": "Absolute", "confidence": 0.98}, ...]
    """

    try:
        response = client.chat_completion(
            model=MODEL_ID,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]}],
            max_tokens=1500
        )
        
        content = response.choices[0].message.content
        clean_json = content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        print(f"❌ Vision Error: {e}")
        return []

def main(photo_file):
    # Step 1: Identification
    card_list = identify_multiple_cards(photo_file)
    
    if not card_list:
        print("⚠️ No cards detected or processing failed.")
        return

    print(f"📦 Detected {len(card_list)} cards. Syncing to database...")

    # Step 2: Validation and Storage
    for card in card_list:
        player = card.get('player', 'Unknown')
        confidence = card.get('confidence', 0)
        
        # Log confidence level
        status = "✅ High Confidence" if confidence >= 0.75 else f"⚠️ LOW CONFIDENCE ({confidence})"
        print(f"{status}: Processing {player}")

        db_entry = {
            "player_name": player,
            "set_name": card.get('pack_set'),
            "card_number": card.get('number'),
            "year": card.get('year')
        }

        try:
            supabase.table("inventory").upsert(db_entry, on_conflict="player_name").execute()
        except Exception as e:
            print(f"❌ Database error for {player}: {e}")

if __name__ == "__main__":
    # Pointing to your specific test file
    main("eightcards.jpg")