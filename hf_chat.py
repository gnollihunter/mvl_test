import os
from huggingface_hub import InferenceClient

TOKEN = os.getenv("HF_TOKEN")

MODELS = {
    "zephyr":  "HuggingFaceH4/zephyr-7b-beta",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "falcon":  "tiiuae/falcon-7b-instruct",
    "gemma":   "google/gemma-2-2b-it",
}

def choose_model():
    print("📦 Verfügbare Modelle:")
    for k, v in MODELS.items():
        print(f"  - {k:7} → {v}")
    raw = input("\nModell wählen (Kurzname oder komplette HF-ID): ").strip()
    return MODELS.get(raw, raw if "/" in raw else MODELS["mistral"])

def ask_chat(client, model_id, history):
    resp = client.chat.completions.create(
        model=model_id,
        messages=history,
        max_tokens=256,
        temperature=0.2,
    )
    return resp.choices[0].message["content"]

def to_prompt(history):
    # simple “instruct”-Prompt aus der History
    sys = next((m["content"] for m in history if m["role"] == "system"), "")
    turns = [m for m in history if m["role"] != "system"]
    prompt = (sys + "\n\n" if sys else "")
    for m in turns:
        if m["role"] == "user":
            prompt += f"User: {m['content']}\n"
        else:
            prompt += f"Assistant: {m['content']}\n"
    prompt += "Assistant:"
    return prompt

def ask_text(client, prompt):
    return client.text_generation(
        prompt,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
        return_full_text=False,
    )

if __name__ == "__main__":
    model_id = choose_model()
    client = InferenceClient(model=model_id, token=TOKEN)
    print(f"✅ Gewählt: {model_id}")

    history = [{"role":"system","content":
        "You are a concise coding assistant. Answer briefly and with runnable code when useful."}]
    print(f"\n🤖 Chat gestartet (exit zum Beenden)\n")

    while True:
        q = input("Du: ")
        if q.strip().lower() in {"exit","quit"}:
            break
        history.append({"role":"user","content": q})

        try:
            ans = ask_chat(client, model_id, history)
        except Exception as e:
            # Fallback auf text-generation
            try:
                ans = ask_text(client, to_prompt(history))
            except Exception as e2:
                print(f"⚠️ Fehler: {type(e2).__name__}: {e2}")
                continue

        print("Bot:", ans)
        history.append({"role":"assistant","content": ans})

