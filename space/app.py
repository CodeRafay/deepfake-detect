"""Hugging Face Space (Gradio + ZeroGPU). For local use, the Streamlit app is deepfake_detect/app.py."""
import gradio as gr
import pandas as pd
import spaces

from deepfake_detect import Detector
from deepfake_detect.faces import draw_faces

# ZeroGPU: load once onto "cuda" at startup; a real GPU is attached only while a @spaces.GPU function runs
det = Detector(device="cuda")

ABOUT = """# deepfake-detect
Upload an image or a video. Every face is scored by a CLIP ViT-L/14 with tuned LayerNorms
([model](https://huggingface.co/CodeRafay/deepfake-detect): 0.914 AUC on unseen Celeb-DF v2).
**A score is evidence, not proof.** Compression, lighting and new kinds of fakes all shift it.
Don't make decisions about people based on this alone. [Code and write-up](https://github.com/CodeRafay/deepfake-detect)"""


def verdict(r):
    return {"fake": r["score"], "real": 1 - r["score"]}


@spaces.GPU(duration=60)
def scan_image(rgb):
    if rgb is None:
        return None, None, None
    r = det.predict_image(rgb)
    return verdict(r), draw_faces(rgb, r), r


@spaces.GPU(duration=120)
def scan_video(path):
    if path is None:
        return None, None, None
    r = det.predict_video(path)
    return verdict(r), pd.DataFrame(r["frames"]), r


with gr.Blocks(title="deepfake-detect") as demo:
    gr.Markdown(ABOUT)
    with gr.Tab("Image"):
        with gr.Row():
            img = gr.Image(type="numpy", label="Image")
            with gr.Column():
                img_label = gr.Label(label="Verdict")
                img_out = gr.Image(label="Faces (red = above threshold)")
        img_json = gr.JSON(label="Raw result")
        img.change(scan_image, img, [img_label, img_out, img_json])
    with gr.Tab("Video"):
        with gr.Row():
            vid = gr.Video(label="Video")
            with gr.Column():
                vid_label = gr.Label(label="Verdict")
                vid_plot = gr.LinePlot(x="time", y="score", y_lim=[0, 1], label="Fake score per sampled frame")
        vid_json = gr.JSON(label="Raw result")
        vid.change(scan_video, vid, [vid_label, vid_plot, vid_json])

demo.launch()
