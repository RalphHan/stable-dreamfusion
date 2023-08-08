import os
import torch
from diffusers import DiffusionPipeline
import gradio as gr
from functools import partial
import uuid

_GPU_INDEX = 0

_TITLE = '''Text-to-3D App built by SeedV Lab'''

_DESCRIPTION = '''
We use SDXL 1.0 to generate image, and use Zero123 to synthesize 3D object
'''

python_path = "venv_stable-dreamfusion/bin/python"


def stage1(base, refiner, prompt):
    image = base(
        prompt=prompt,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        image=image,
    ).images[0]
    os.makedirs("data/gradio", exist_ok=True)
    the_uuid = str(uuid.uuid4())
    image.save(f"data/gradio/{the_uuid}.png")
    return image, the_uuid


def stage2(image, the_uuid):
    if the_uuid == "":
        os.makedirs("data/gradio", exist_ok=True)
        the_uuid = str(uuid.uuid4())
        image.save(f"data/gradio/{the_uuid}.png")
    os.makedirs(".tasks/running", exist_ok=True)
    os.system(f"touch .tasks/running/{the_uuid}")
    os.system(f"{python_path} preprocess_image.py data/gradio/{the_uuid}.png")
    return f"data/gradio/{the_uuid}_rgba.png", the_uuid


def stage3(h_w, iters, the_uuid):
    if os.path.exists(f".tasks/stop/{the_uuid}"):
        os.remove(f".tasks/stop/{the_uuid}")
        raise Exception("Stopped")
    ret_code = os.system(f"UUID={the_uuid} {python_path} main.py -O --h {h_w} --w {h_w} "
                         f"--image data/gradio/{the_uuid}_rgba.png "
                         f"--workspace results/gradio/{the_uuid} --iters {iters} --save_mesh")
    if ret_code != 0:
        os.remove(f".tasks/stop/{the_uuid}")
        raise Exception("Error in stage3")
    video_name = sorted([x for x in os.listdir(f"results/gradio/{the_uuid}/results/") if x.endswith("_rgb.mp4")])[-1]
    full_video_name = os.path.join(f"results/gradio/{the_uuid}/results/", video_name)
    return full_video_name


def stage4(h_w, iters, the_uuid):
    if os.path.exists(f".tasks/stop/{the_uuid}"):
        os.remove(f".tasks/stop/{the_uuid}")
        raise Exception("Stopped")
    ret_code = os.system(f"UUID={the_uuid} {python_path} main.py -O --h {h_w} --w {h_w} "
                         f"--image data/gradio/{the_uuid}_rgba.png "
                         f"--workspace results/gradio/{the_uuid}_dmtet --iters {iters} --save_mesh "
                         f"--dmtet --init_with results/gradio/{the_uuid}/checkpoints/df.pth")
    if ret_code != 0:
        os.remove(f".tasks/stop/{the_uuid}")
        raise Exception("Error in stage4")
    os.system(f"rm .tasks/running/{the_uuid}")
    video_name = sorted([x for x in os.listdir(f"results/gradio/{the_uuid}_dmtet/results/") if x.endswith("_rgb.mp4")])[
        -1]
    full_video_name = os.path.join(f"results/gradio/{the_uuid}_dmtet/results/", video_name)
    os.system(f"cd results/gradio/{the_uuid}_dmtet/mesh/ && zip -v mesh.zip *")
    return full_video_name, f"results/gradio/{the_uuid}_dmtet/mesh/mesh.zip"


def kill_all():
    os.makedirs(".tasks/stop", exist_ok=True)
    os.system("mv .tasks/running/* .tasks/stop/")
    os.system("pkill -9 stable-dreamfusion")


def kill(the_uuid):
    if the_uuid and os.path.exists(f".tasks/running/{the_uuid}"):
        os.makedirs(".tasks/stop", exist_ok=True)
        os.system(f"mv .tasks/running/{the_uuid} .tasks/stop/")
        os.system(f"pkill -9 stable-dreamfusion-{the_uuid}")


def run_demo():
    device = f"cuda:{_GPU_INDEX}"
    os.system("rm -rf .tasks")
    # load both base & refiner
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
    # Compose demo layout & data flow.
    css = "#model-3d-out {height: 400px;} #plot-out {height: 450px;}"
    with gr.Blocks(title=_TITLE, css=css).queue(concurrency_count=1) as demo:
        gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)
        the_uuid = gr.State("")
        with gr.Row():
            prompt = gr.Text("zuckerberg, full body, blender 3d, "
                             "artstation and behance, Disney Pixar, "
                             "smooth lighting, smooth background color, "
                             "(front view:1.5), cute, hd, 8k", label="Prompt", lines=5)
            btn_sdxl = gr.Button("Generate Image", variant="secondary")
            image = gr.Image(type='pil', label='Generated or Input image')
        with gr.Row():
            with gr.Column():
                h_w = gr.Dropdown(choices=[32, 64, 128], value=64, label="h_w")
                iters = gr.Slider(minimum=1, maximum=15000, step=1000, value=5000, label="iters")
                iters_dmtet = gr.Slider(minimum=1, maximum=15000, step=1000, value=10000, label="iters_dmtet")
            with gr.Column():
                btn_dreamfusion = gr.Button("Generate 3D Object", variant="primary")
                btn_kill = gr.Button("Kill (remember to refresh the browser after clicking)", variant="stop")
                btn_kill_all = gr.Button("Kill All", variant="stop")

            image_nobg = gr.Image(image_mode='RGBA', label='Remove BG')
        with gr.Row():
            video = gr.Video(format="mp4", autoplay=True)
            video_dmtet = gr.Video(format="mp4", autoplay=True)
            mesh = gr.File(label="Download 3D Object")

        btn_sdxl.click(fn=partial(stage1, base, refiner), inputs=[prompt], outputs=[image, the_uuid], queue=False)
        btn_dreamfusion.click(fn=stage2, inputs=[image, the_uuid], outputs=[image_nobg, the_uuid], queue=False
                              ).success(fn=stage3, inputs=[h_w, iters, the_uuid], outputs=[video]
                                        ).success(fn=stage4, inputs=[h_w, iters_dmtet, the_uuid],
                                                  outputs=[video_dmtet, mesh]
                                                  )
        btn_kill.click(fn=kill, inputs=[the_uuid], queue=False)
        btn_kill_all.click(fn=kill_all, queue=False)

    demo.launch(enable_queue=True, share=True, max_threads=1)


if __name__ == '__main__':
    run_demo()
