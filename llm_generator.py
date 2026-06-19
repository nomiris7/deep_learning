import json
import requests


def generate_architecture_from_llm(model_name="llama3"):
    prompt = """
    Generate a CNN architecture for CIFAR-10. 
    Allowed layers: Conv2D, MaxPool, Dropout, Linear, BatchNorm, ReLU, GlobalAveragePooling.
    Max 6 convolutional layers. 
    Return ONLY valid JSON format exactly like this example, with no markdown formatting or extra text:
    {"layers": [{"type":"conv","filters":32,"kernel":3}, {"type":"maxpool"}]}
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.8},
            },
        )
        response.raise_for_status()

        # Oczyszczanie odpowiedzi i parsowanie JSON
        raw_text = response.json()["response"].strip()
        return json.loads(raw_text)
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"[LLM Error] Nie udało się wygenerować poprawnej architektury: {e}")
        return None
