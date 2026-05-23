import torch
import gc
from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from audiocraft.models import MusicGen
import scipy.io.wavfile as wavfile

# Konfiguracja urządzeń
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


def flush_memory():
    """Agresywne czyszczenie VRAM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print(f"Uruchamianie lekkiego pipeline na urządzeniu: {device}")

# ==========================================
# KROK 1: Napisz prompt
# ==========================================
initial_prompt = "Small university in Cracow, located near church, name is WSEI and main colors are green and white. It is mainly IT and economics"

# ==========================================
# KROK 2: Wygeneruj obraz (Lżejszy model SD 1.5)
# ==========================================
print("\n--- Krok 2: Generowanie pierwszego obrazu (SD 1.5) ---")
with torch.no_grad():
    image_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=dtype
    ).to(device)

    # Dodatkowa optymalizacja dla kart 8GB
    if device == "cuda":
        image_pipe.enable_attention_slicing()

    image_1 = image_pipe(prompt=initial_prompt, num_inference_steps=25).images[0]
    image_1.save("wsei_obraz_1.png")

# Natychmiastowe usunięcie modelu z VRAM
image_pipe = image_pipe.to("cpu")
del image_pipe
flush_memory()

# ==========================================
# KROK 3: Opis obrazu (Lżejszy model BLIP Base)
# ==========================================
print("\n--- Krok 3: Generowanie opisu z obrazu 1 (BLIP Base) ---")
with torch.no_grad():
    vlm_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    vlm_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", torch_dtype=dtype
    ).to(device)

    inputs = vlm_processor(images=image_1, return_tensors="pt").to(device, dtype)
    generated_ids = vlm_model.generate(**inputs, max_new_tokens=50)
    generated_description = vlm_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()

print(f"Wygenerowany opis: {generated_description}")

# Czyszczenie po VLM
vlm_model = vlm_model.to("cpu")
del vlm_model, vlm_processor
flush_memory()

# ==========================================
# KROK 4: Wygeneruj nowy obrazek na podstawie opisu z kroku 3
# ==========================================
print("\n--- Krok 4: Generowanie drugiego obrazu na podstawie nowego opisu ---")
with torch.no_grad():
    image_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=dtype
    ).to(device)
    if device == "cuda":
        image_pipe.enable_attention_slicing()

    final_image_prompt = (
        f"{generated_description}, futuristic campus, architectural photography"
    )
    image_2 = image_pipe(prompt=final_image_prompt, num_inference_steps=25).images[0]
    image_2.save("wsei_obraz_2.png")

image_pipe = image_pipe.to("cpu")
del image_pipe
flush_memory()

# ==========================================
# KROK 5: Wygeneruj muzykę (Model SMALL zamiast Medium)
# ==========================================
print("\n--- Krok 5: Generowanie muzyki (MusicGen Small) ---")
with torch.no_grad():
    # Pobieramy wersję 'small' (zajmuje ułamek pamięci wersji medium)
    music_model = MusicGen.get_pretrained("facebook/musicgen-small")
    music_model.set_generation_params(duration=7)  # 7 sekund, żeby oszczędzić ram

    music_prompt = (
        "Cyberpunk lo-fi coding music, electronics, university hackathon theme, 110 BPM"
    )
    wav_output = music_model.generate([music_prompt])

    sampling_rate = music_model.sample_rate
    wavfile.write(
        "wsei_music.wav", rate=sampling_rate, data=wav_output[0, 0].cpu().numpy()
    )

# Czyszczenie muzyki
del music_model
flush_memory()

# ==========================================
# KROK 6: Music to Image (Generowanie obrazu na podstawie muzyki)
# ==========================================
print("\n--- Krok 6: Generowanie obrazu na podstawie muzyki ---")
with torch.no_grad():
    image_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=dtype
    ).to(device)
    if device == "cuda":
        image_pipe.enable_attention_slicing()

    music_to_image_prompt = "Abstract digital soundwaves illuminating a dark computer science room, retro cybernetic aesthetic"
    image_3 = image_pipe(prompt=music_to_image_prompt, num_inference_steps=25).images[0]
    image_3.save("wsei_obraz_z_muzyki.png")

image_pipe = image_pipe.to("cpu")
del image_pipe
flush_memory()

# ==========================================
# KROK 7: Opis końcowego obrazka z kroku 6
# ==========================================
print("\n--- Krok 7: Finalny opis obrazu wygenerowanego z muzyki ---")
with torch.no_grad():
    vlm_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    vlm_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", torch_dtype=dtype
    ).to(device)

    inputs_final = vlm_processor(images=image_3, return_tensors="pt").to(device, dtype)
    generated_ids_final = vlm_model.generate(**inputs_final, max_new_tokens=50)
    final_description = vlm_processor.batch_decode(
        generated_ids_final, skip_special_tokens=True
    )[0].strip()

vlm_model = vlm_model.to("cpu")
del vlm_model, vlm_processor
flush_memory()

print("\n=== PIPELINE UKOŃCZONY POMYŚLNIE ===")
print(f"Początkowy koncept: {initial_prompt}")
print(f"Krok 3 (Opis pośredni): {generated_description}")
print(f"Krok 7 (Opis finalny z muzyki): {final_description}")
