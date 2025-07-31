# ─── 1. 基底映像：Ubuntu 20.04 + CUDA 12.1.1 + cuDNN8 ────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

# ─── 2. 安裝系統套件 ─────────────────────────────────────────────────────────
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      wget bzip2 ca-certificates git libglib2.0-0 libxext6 libsm6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# ─── 3. 安裝 Miniconda ───────────────────────────────────────────────────────
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
 && rm /tmp/miniconda.sh \
 && conda clean -afy

WORKDIR /app

# ─── 4. 複製並建立 Conda 環境 ───────────────────────────────────────────────
COPY environment.yml .
RUN conda config --set always_yes yes --set changeps1 no \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
 && conda env create -f environment.yml \
 && conda clean -afy

# ─── 5. 設定環境變數以啟用 variant 環境 ────────────────────────────────────
ENV PATH=/opt/conda/envs/variant/bin:$PATH
ENV CONDA_DEFAULT_ENV=variant

# ─── 6. 下載並配置 NLTK punkt 分句模型 ───────────────────────────────────────
RUN python - <<'EOF'
import nltk, os
# 下載 punkt 模型到指定目錄
nltk.download('punkt', download_dir='/opt/conda/envs/variant/nltk_data')

# 建立 punkt_tab/english 並建立軟連結
root = '/opt/conda/envs/variant/nltk_data/tokenizers'
src = os.path.join(root, 'punkt', 'english.pickle')
dst_dir = os.path.join(root, 'punkt_tab', 'english')
os.makedirs(dst_dir, exist_ok=True)
if not os.path.exists(os.path.join(dst_dir, 'english.pickle')):
    os.symlink(src, os.path.join(dst_dir, 'english.pickle'))
EOF

# ─── 7. 複製應用程式原始碼與模型檔案 ─────────────────────────────────────────
# 請確保專案根目錄包含 run_app.py、pubtator/、outputs/、config.yaml 等
COPY . .

# ─── 8. 暴露埠口並啟動應用程式 ────────────────────────────────────────
EXPOSE 8080
ENTRYPOINT ["python", "run_app.py"]
