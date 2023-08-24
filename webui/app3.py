import os
import gradio as gr
import uuid

python_path = "venv_stable-dreamfusion/bin/python"


def run(prompt):
    os.makedirs("data/gradio", exist_ok=True)
    the_uuid = str(uuid.uuid4())
    image_path = f"data/gradio/{the_uuid}.png"
    os.system(f"{python_path} sdxl_infer.py \"{prompt}\" {image_path}")
    return image_path


def run_demo():
    css = "#model-3d-out {height: 400px;} #plot-out {height: 450px;}"
    with gr.Blocks(css=css).queue(concurrency_count=1) as demo:
        with gr.Row():
            prompt = gr.Text("zuckerberg, full body, blender 3d, "
                             "artstation and behance, Disney Pixar, "
                             "smooth lighting, smooth background color, "
                             "(front view:1.5), cute, hd, 8k", label="Prompt", lines=5)
            btn_run = gr.Button("Generate", variant="primary")
        with gr.Row():
            image = gr.Image(type='pil', label='Generated Image')
        btn_run.click(fn=run, inputs=[prompt], outputs=[image])
    demo.launch(server_name="0.0.0.0")


if __name__ == '__main__':
    run_demo()
