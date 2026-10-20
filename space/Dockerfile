FROM python:3.11-slim

# git: pip install from GitHub; libgl1/libglib: OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "deepfake-detect[app] @ git+https://github.com/CodeRafay/deepfake-detect"

# Download CLIP, the latest head and the face detector at build time, so the first visitor doesn't wait
RUN python -c "from deepfake_detect import Detector; from deepfake_detect.faces import _detector; Detector(device='cpu'); _detector()"

EXPOSE 7860
# XSRF off: Spaces serve the app in an iframe, where Streamlit's XSRF check rejects file uploads
CMD streamlit run "$(python -c 'import deepfake_detect, os; print(os.path.dirname(deepfake_detect.__file__))')/app.py" \
    --server.port 7860 --server.address 0.0.0.0 --server.enableXsrfProtection false
