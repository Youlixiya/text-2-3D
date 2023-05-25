import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
#stable diffusion
def load_img2img_pipeline(sd_model_key):
    global img2img_pipeline
    img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(sd_model_key, torch_dtype=torch.float16).to(device)
    return 'stable_diffusion img2img_pipeline loads!'
def clear_img2img_pipeline():
    global img2img_pipeline
    del img2img_pipeline
    torch.cuda.empty_cache()
    return 'img2img_pipeline clears!'
def load_text2img_pipeline(sd_model_key):
    global text2img_pipeline
    text2img_pipeline = StableDiffusionPipeline.from_pretrained(sd_model_key, torch_dtype=torch.float16).to(device)
    return 'stable_diffusion text2img_pipeline loads!'
def clear_text2img_pipeline():
    global text2img_pipeline
    del text2img_pipeline
    torch.cuda.empty_cache()
    return 'text2img_pipeline clears!'
def sd_img2img(image, prompt, save_image_name, strength, guidance_scale):
    image = image.resize((768, 512))
    image = img2img_pipeline(prompt=prompt, image=image, strength=eval(strength), guidance_scale=eval(guidance_scale)).images[0]
    image.save(f"images/{save_image_name}_img2img.png")
    return image
def sd_text2img(prompt, save_image_name):
    image = text2img_pipeline(prompt=prompt).images[0]
    image.save(f"images/{save_image_name}_text2img.png")
    return image