import torch
from diffusers import DiffusionPipeline
from setproctitle import setproctitle
setproctitle('StableDream')
import sys

prompt = sys.argv[1]
save_path = sys.argv[2]
device = "cuda"
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to(device)
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to(device)
image = base(
    prompt=prompt,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    image=image,
).images[0]

image.save(save_path)
