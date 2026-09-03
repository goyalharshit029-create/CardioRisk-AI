import os
from pathlib import Path

from google import genai


# ============================================================
# LOAD .env FILE
# ============================================================

ROOT = Path(__file__).resolve().parent

ENV_PATH = ROOT / ".env"


if ENV_PATH.exists():

    print(".env file found.")

    for line in ENV_PATH.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if (
            line
            and not line.startswith("#")
            and "=" in line
        ):

            key, value = line.split("=", 1)

            os.environ[key.strip()] = (
                value.strip()
                .strip('"')
                .strip("'")
            )

else:

    print("❌ .env file NOT found.")


# ============================================================
# GET API KEY
# ============================================================

API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
).strip()


print(
    "API Key found:",
    bool(API_KEY)
)


if not API_KEY:

    print("\n❌ GEMINI_API_KEY not found!")

    print(
        "Make sure your backend/.env file contains:"
    )

    print(
        "GEMINI_API_KEY=your_actual_api_key"
    )

    exit()


# ============================================================
# CREATE CLIENT
# ============================================================

print("\n🤖 Creating Gemini client...")

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# TEST GEMINI
# ============================================================

print("💬 Sending test request...")


try:

    response = client.models.generate_content(

        model="gemini-3.6-flash",

        contents="""
Mera moderate cardiovascular risk hai.
Mujhe Hinglish mein batao ki main fit kaise rahu.
Keep the answer practical and concise.
"""

    )


    print("\n✅ RESPONSE RECEIVED:\n")

    print(response.text)


except Exception as e:

    print("\n❌ ERROR:\n")

    print(type(e).__name__)

    print(e)