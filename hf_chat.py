import os
from huggingface_hub import InferenceClient

# ---- Modelle (kannst du frei erweitern) ------------------------------
MODELS = {
    "zephyr":  ("HuggingFaceH4/zephyr-7b-beta", "chat"),   # conversational
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.2", "text"),
    "falcon":  ("tiiuae/falcon-7b-instruct", "text"),
    "gemma":   ("google/gemma-2-2b-it", "text"),
}
# ---------------------------------------------------------------------

TOKEN = os.getenv("HF_TOKEN")

def choose_model():
    print("📦 Verfügbare Modelle:")
    for key, (mid, mode) in MODELS.items():
        print(f"  - {key:7} → {mid}  [{mode}]")
    raw = input("\nModell wählen (Kurzname oder komplette HF-ID): ").strip()

    if raw in MODELS:
        return MODELS[raw]
    # komplette HF-ID?
    if "/" in raw:
        # Standard: wir nehmen 'text' – bei Bedarf einfach auf 'chat' ändern
        return (raw, "text")
    # Fallback: Mistral
    print("Unbekannt – nehme 'mistral'.")
    return MODELS["mistral"]

def run_chat(model_id: str):
    client = InferenceClient(model=model_id, token=TOKEN)
    history = [{"role": "system",
                "content": "You are a concise coding assistant. Answer briefly and with runnable code when useful."}]
    print(f"\n🤖 Chat auf {model_id}  (exit zum Beenden)")
    while True:
        q = input("\nDu: ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        history.append({"role": "user", "content": q})
        try:
            resp = client.chat.completions.create(
                model=model_id, messages=history, max_tokens=256, temperature=0.2
            )
            ans = resp.choices[0].message["content"]
            print("Bot:", ans)
            history.append({"role": "assistant", "content": ans})
        except Exception as e:
            print("⚠️ Fehler (chat):", e)

def run_text(model_id: str):
    client = InferenceClient(model=model_id, token=TOKEN)
    print(f"\n📝 Text-Gen auf {model_id}  (exit zum Beenden)")
    while True:
        q = input("\nDu: ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        prompt = f"You are a concise coding assistant. Answer briefly.\n\nUser: {q}\nAssistant:"
        try:
            for chunk in client.text_generation(
                prompt, max_new_tokens=256, temperature=0.2,
                return_full_text=False, stream=True
            ):
                print(chunk, end="", flush=True)
            print()
        except Exception as e:
            print("⚠️ Fehler (text):", e)

if __name__ == "__main__":
    model_id, mode = choose_model()
    print(f"✅ Gewählt: {model_id} [{mode}]")
    if mode == "chat":
        run_chat(model_id)
    else:
        run_text(model_id)
