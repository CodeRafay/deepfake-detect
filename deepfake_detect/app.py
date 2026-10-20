"""Web demo: streamlit run deepfake_detect/app.py"""
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from deepfake_detect.faces import IMAGE_EXTS, VIDEO_EXTS, draw_faces
from deepfake_detect.model import Detector

st.set_page_config(page_title="deepfake-detect", page_icon="🔍", layout="wide")


@st.cache_resource
def detector():
    return Detector()


st.title("deepfake-detect")
st.caption("Finds faces, scores each with a frozen CLIP encoder + linear probe, and aggregates. "
           "**A score is evidence, not proof**: heavy compression, unusual lighting and generators the model "
           "never saw all shift scores. Don't make decisions about people based on this alone.")

up = st.file_uploader("Image or video", type=sorted(e[1:] for e in IMAGE_EXTS | VIDEO_EXTS))
if up:
    suffix = Path(up.name).suffix.lower()
    left, right = st.columns(2)
    with st.spinner("Scanning..."):
        if suffix in VIDEO_EXTS:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(up.getvalue())
            try:
                r = detector().predict_video(f.name)
            finally:
                os.unlink(f.name)
            left.video(up)
            right.line_chart(pd.DataFrame(r["frames"]).set_index("time")["score"], y_label="fake score")
        else:
            bgr = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                st.error("Could not read that image.")
                st.stop()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            r = detector().predict_image(rgb)
            left.image(rgb, caption="Input")
            right.image(draw_faces(rgb, r), caption="Faces (red = above threshold)")

    st.metric("Verdict", r["label"].upper(), f"score {r['score']:.3f} · threshold {r['threshold']:.3f}",
              delta_color="off")
    if "note" in r:
        st.warning(r["note"])
    with st.expander("Raw result"):
        st.json(r)
