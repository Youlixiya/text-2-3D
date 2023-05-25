import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
device = "cuda" if torch.cuda.is_available() else "cpu"
#BLIP

def load_blip(blip_model_key):
    global blip_processor, blip
    blip_processor = BlipProcessor.from_pretrained(blip_model_key)
    blip = BlipForConditionalGeneration.from_pretrained(
        blip_model_key, torch_dtype=torch.float16
    )
    blip.to(device)
    return 'blip loads!'

def clear_blip():
    global blip_processor, blip
    del blip_processor, blip
    torch.cuda.empty_cache()
    return 'blip clears!'

def blip_generate_caption(image):
    text = "a photography of"
    inputs = blip_processor(images=image, text=text, return_tensors="pt").to(device, torch.float16)
    generated_ids = blip.generate(**inputs)
    generated_text = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text