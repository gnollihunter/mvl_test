import os
from huggingface_hub import InferenceClient

MODEL = "HuggingFaceH4/zephyr-7b-beta"
TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(model=MODEL, token=TOKEN)

print(f"🤖 HF Chat auf {MODEL} (exit zum Beenden)") #hier wird das Modell gewählt, hab aber nicht verstande, wie!
print("🔧 Version 2: jetzt mit Teständerung!")
history = [{"role": "system", "content": "You are a helpful coding assistant. Answer briefly."}]

while True:
    q = input("\nDu: ")
    if q.strip().lower() in {"exit", "quit"}:
        break

    history.append({"role": "user", "content": q})
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=history,
            max_tokens=256,
            temperature=0.2,
        )
        answer = resp.choices[0].message["content"]
        print("Bot:", answer)
        history.append({"role": "assistant", "content": answer})
    except Exception as e:
        print("⚠️ Fehler:", e)
        
