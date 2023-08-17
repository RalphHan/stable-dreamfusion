import os
import gradio as gr
import uuid

_TITLE = '''Text-to-3D App built by SeedV Lab'''

python_path = "venv_stable-dreamfusion/bin/python"


def stage1(prompt, the_uuid):
    os.makedirs("data/gradio", exist_ok=True)
    image_path = f"data/gradio/{the_uuid}.png"
    ret_code = os.system(f"{python_path} sdxl_infer.py \"{prompt}\" {image_path}")
    assert ret_code == 0, "Error in stage1"


def stage2(the_uuid):
    assert the_uuid != "", "Error in stage2"
    ret_code = os.system(f"{python_path} preprocess_image.py data/gradio/{the_uuid}.png")
    assert ret_code == 0, "Error in stage2"


def stage3(h_w, iters, the_uuid):
    ret_code = os.system(f"{python_path} main.py -O --h {h_w} --w {h_w} "
                         f"--image data/gradio/{the_uuid}_rgba.png "
                         f"--workspace results/gradio/{the_uuid} --iters {iters} --save_mesh")
    assert ret_code == 0, "Error in stage3"


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
    os.makedirs("results/gradio_bk", exist_ok=True)
    os.system(f"mv results/gradio/{the_uuid}_dmtet/mesh/mesh.zip results/gradio_bk/{the_uuid}_dmtet_mesh.zip")
    return full_video_name, f"results/gradio_bk/{the_uuid}_dmtet_mesh.zip"


def run(prompt, h_w, iters, iters_dmtet):
    the_uuid = str(uuid.uuid4())
    prompt = prompt + ", full body, blender 3d, " \
                      "artstation and behance, Disney Pixar, " \
                      "smooth lighting, smooth background color, " \
                      "(front view:1.5), cute, hd, 8k"
    stage1(prompt, the_uuid)
    stage2(the_uuid)
    stage3(h_w, iters, the_uuid)
    video_name, mesh_name = stage4(h_w, iters_dmtet, the_uuid)
    return video_name, mesh_name


def kill_all():
    os.system("pkill -9 StableDream")
    os.system("rm -rf results/gradio")


def run_demo():
    # Compose demo layout & data flow.
    css = "#model-3d-out {height: 400px;} #plot-out {height: 450px;}"
    with gr.Blocks(title=_TITLE, css=css).queue(concurrency_count=1) as demo:
        gr.Markdown('# ' + _TITLE)
        with gr.Row():
            with gr.Column():
                prompt = gr.Text("zuckerberg", label="Prompt")
                with gr.Accordion('Advanced options', open=False):
                    h_w = gr.Dropdown(choices=["32", "64", "128", "256"], value="64", label="h_w")
                    iters = gr.Slider(minimum=1, maximum=15000, step=1000, value=5000, label="iters")
                    iters_dmtet = gr.Slider(minimum=1, maximum=15000, step=1000, value=5000, label="iters_dmtet")
            with gr.Column():
                btn_run = gr.Button("Generate 3D Object", variant="primary")
                btn_kill_all = gr.Button("Kill All", variant="stop")
        with gr.Row():
            with gr.Column(scale=0.5):
                video = gr.Video(format="mp4", label="Video", autoplay=True, width=200, height=200)
            with gr.Column(scale=0.5):
                mesh = gr.File(label="Download 3D Object")
        btn_run.click(fn=run, inputs=[prompt, h_w, iters, iters_dmtet], outputs=[video, mesh])
        btn_kill_all.click(fn=kill_all, queue=False)

    demo.launch(server_name="0.0.0.0")


if __name__ == '__main__':
    run_demo()
