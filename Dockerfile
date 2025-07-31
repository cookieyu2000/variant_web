# ─── 1. Base: CUDA 12.1 + cuDNN8 on Ubuntu 20.04 ────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

# ─── 2. 系统依赖 ───────────────────────────────────────────────────────
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      wget bzip2 ca-certificates git libglib2.0-0 libxext6 libsm6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# ─── 3. 安装 Miniconda ──────────────────────────────────────────────────
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
 && rm /tmp/miniconda.sh \
 && conda clean -afy

WORKDIR /app

# ─── 4. 复制并创建 Conda 环境 ──────────────────────────────────────────
COPY environment.yml .
RUN conda config --set always_yes yes --set changeps1 no \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
 && conda env create -f environment.yml \
 && conda clean -afy

# ─── 5. 切换到 variant 环境 ────────────────────────────────────────────
ENV PATH=/opt/conda/envs/variant/bin:$PATH
ENV CONDA_DEFAULT_ENV=variant

# ─── 6. 下载 NLTK punkt 并手动放到 punkt_tab/english ───────────────────
RUN python - <<'EOF'
import nltk, os
# 下载 punkt 模型
nltk.download('punkt', download_dir='/opt/conda/envs/variant/nltk_data')

root = '/opt/conda/envs/variant/nltk_data/tokenizers'
src = os.path.join(root, 'punkt', 'english.pickle')
dst_dir = os.path.join(root, 'punkt_tab', 'english')
os.makedirs(dst_dir, exist_ok=True)
# 将 english.pickle 软链接到 punkt_tab/english/english.pickle
if not os.path.exists(os.path.join(dst_dir, 'english.pickle')):
    os.symlink(src, os.path.join(dst_dir, 'english.pickle'))
EOF

# ─── 7. 复制应用代码 & 模型输出 ──────────────────────────────────────────
# 确保项目根包括 run_app.py、pubtator/、outputs/、config.yaml 等
COPY . .

# ─── 8. 暴露端口 & 启动 ────────────────────────────────────────────────
EXPOSE 8080
ENTRYPOINT ["python", "run_app.py"]
