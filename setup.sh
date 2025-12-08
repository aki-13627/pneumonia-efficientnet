#!/bin/bash

# --- 0. 変数設定 ---
REPO_URL="https://github.com/aki-13627/pneumonia-efficientnet.git"
REPO_DIR="pneumonia-efficientnet"

# --- 1. Gitリポジトリのクローン  ---
echo "--- 1. Cloning repository ---"
if [ -d "$REPO_DIR" ]; then
    echo "Repository already exists. Pulling latest changes..."
    cd $REPO_DIR
    git pull
else
    git clone $REPO_URL
    cd $REPO_DIR
fi

# --- 2. Python環境セットアップ  ---
echo "--- 2. Setting up Python environment ---"
sudo apt update
sudo apt install -y python3-pip python3-venv 

python3 -m venv venv_gpu
source venv_gpu/bin/activate

# --- 3. ライブラリのインストール  ---
echo "Installing Python dependencies (TensorFlow, Scikit-learn, etc.)..."
pip install -r requirements.txt

# --- 4. GPU認識の確認  ---
echo "--- 4. Verifying TensorFlow GPU access (Optional) ---"
./venv_gpu/bin/python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU devices:', tf.config.list_physical_devices('GPU'))"

# --- 5. 学習の開始 ---
echo "--- 5. Starting MobileNetV2 HPO Training in Background ---"

# 現在時刻を取得し、ファイル名に組み込む
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_output_${TIMESTAMP}.log"

# nohupを使ってバックグラウンドで実行し、出力をログファイルにリダイレクト
nohup ./venv_gpu/bin/python3 models/train.py > $LOG_FILE 2>&1 &

echo "学習をバックグラウンドで開始しました。"
echo "ログファイル名: $LOG_FILE"
echo "進捗確認は 'tail -f $LOG_FILE' で行ってください。"
echo "PID: $!"

echo "Setup and Training Script Finished."