import os
import gradio as gr
import uuid

_TITLE = '''Text-to-3D App built by SeedV Lab'''

_DESCRIPTION = '''
We use SDXL 1.0 to generate image, and use Zero123 to synthesize 3D object
'''

python_path = "venv_stable-dreamfusion/bin/python"


def stage1(prompt):
    os.makedirs("data/gradio", exist_ok=True)
    the_uuid = str(uuid.uuid4())
    save_path = f"data/gradio/{the_uuid}.png"
    os.system(f"{python_path} sdxl_infer.py \"{prompt}\" {save_path}")
    return save_path, the_uuid


def stage2(image, the_uuid):
    if the_uuid == "":
        the_uuid = str(uuid.uuid4())
    os.makedirs("data/gradio", exist_ok=True)
    image.save(f"data/gradio/{the_uuid}.png")
    ret_code = os.system(f"{python_path} preprocess_image.py data/gradio/{the_uuid}.png")
    assert ret_code == 0, "Error in stage2"
    return f"data/gradio/{the_uuid}_rgba.png", the_uuid


def stage3(h_w, iters, the_uuid):
    ret_code = os.system(f"{python_path} main.py -O --h {h_w} --w {h_w} "
                         f"--image data/gradio/{the_uuid}_rgba.png "
                         f"--workspace results/gradio/{the_uuid} --iters {iters} --save_mesh")
    assert ret_code == 0, "Error in stage3"
    video_name = sorted([x for x in os.listdir(f"results/gradio/{the_uuid}/results/") if x.endswith("_rgb.mp4")])[-1]
    full_video_name = os.path.join(f"results/gradio/{the_uuid}/results/", video_name)
    return full_video_name


def stage4(h_w, iters, the_uuid):
    ret_code = os.system(f"{python_path} main.py -O --h {h_w} --w {h_w} "
                         f"--image data/gradio/{the_uuid}_rgba.png "
                         f"--workspace results/gradio/{the_uuid}_dmtet --iters {iters} --save_mesh "
                         f"--dmtet --init_with results/gradio/{the_uuid}/checkpoints/df.pth")
    assert ret_code == 0, "Error in stage4"
    video_name = sorted([x for x in os.listdir(f"results/gradio/{the_uuid}_dmtet/results/") if x.endswith("_rgb.mp4")])[
        -1]
    full_video_name = os.path.join(f"results/gradio/{the_uuid}_dmtet/results/", video_name)
    os.system(f"cd results/gradio/{the_uuid}_dmtet/mesh/ && zip -v mesh.zip *")
    return full_video_name, f"results/gradio/{the_uuid}_dmtet/mesh/mesh.zip"


def kill_all():
    os.system("pkill -9 StableDream")


def run_demo():
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
            image = gr.Image(type='pil', label='Generated or Input image')
        with gr.Row():
            with gr.Column():
                h_w = gr.Dropdown(choices=["32", "64", "128", "256"], value="64", label="h_w")
                iters = gr.Slider(minimum=1, maximum=15000, step=1000, value=5000, label="iters")
                iters_dmtet = gr.Slider(minimum=1, maximum=15000, step=1000, value=10000, label="iters_dmtet")
            with gr.Column():
                btn_sdxl = gr.Button("Generate Image", variant="secondary")
                btn_dreamfusion = gr.Button("Generate 3D Object", variant="primary")
                btn_kill_all = gr.Button("Kill All", variant="stop")

            image_nobg = gr.Image(image_mode='RGBA', label='Remove BG')
        with gr.Row():
            video = gr.Video(format="mp4", autoplay=True)
            video_dmtet = gr.Video(format="mp4", autoplay=True)
            mesh = gr.File(label="Download 3D Object")

        btn_sdxl.click(fn=stage1, inputs=[prompt], outputs=[image, the_uuid], queue=False)
        btn_dreamfusion.click(lambda: gr.update(interactive=False), outputs=btn_sdxl, queue=False
                              ).success(fn=stage2, inputs=[image, the_uuid], outputs=[image_nobg, the_uuid]
                                        ).success(fn=stage3, inputs=[h_w, iters, the_uuid], outputs=[video]
                                                  ).success(fn=stage4, inputs=[h_w, iters_dmtet, the_uuid],
                                                            outputs=[video_dmtet, mesh]
                                                            ).success(fn=lambda: gr.update(interactive=True),
                                                                      outputs=btn_sdxl,
                                                                      queue=False
                                                                      )
        btn_kill_all.click(fn=kill_all, queue=False)

    demo.launch(share=True)


if __name__ == '__main__':
    run_demo()
