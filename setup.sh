#!/bin/bash

# --- 0. 変数設定 ---
REPO_URL="https://github.com/aki-13627/pneumonia-efficientnet.git"
REPO_DIR="pneumonia-efficientnet"

# --- 1. Gitリポジトリのクローン ---
echo "--- 1. Cloning repository ---"
if [ -d "$REPO_DIR" ]; then
    echo "Repository already exists. Pulling latest changes..."
    cd $REPO_DIR
    git pull
else
    git clone $REPO_URL
    cd $REPO_DIR
fi

# --- 2. Python環境セットアップ ---
echo "--- 2. Setting up Python environment ---"
# aptコマンドの更新と必要なツールのインストール (pip, venv, opencv dependencies)
sudo apt update
# python3-opencv は競合する場合があるため、pipに任せる。必要な基本パッケージをインストール
sudo apt install -y python3-pip python3-venv 

# 仮想環境の作成とアクティベート
python3 -m venv venv_gpu
source venv_gpu/bin/activate

# --- 3. ライブラリのインストール ---
# GPU版TensorFlowと必要なライブラリをインストール
echo "Installing Python dependencies (TensorFlow, Scikit-learn, etc.)..."
pip install -r requirements.txt

# --- 4. GPU認識の確認 ---
echo "--- 4. Verifying TensorFlow GPU access (Optional) ---"
# venv_gpu のPythonを使って確認
./venv_gpu/bin/python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU devices:', tf.config.list_physical_devices('GPU'))"

# --- 5. 学習の開始 ---
echo "--- 5. Starting MobileNetV2 HPO Training in Background ---"

# ログファイル名を設定
LOG_FILE="hpo_training_output.log"

# nohupを使ってバックグラウンドで実行し、出力をログファイルにリダイレクト
# (& をつけることで、即座に制御がシェルに戻る)
nohup ./venv_gpu/bin/python3 utils/hpo.py > $LOG_FILE 2>&1 &

echo "HPOプロセスをバックグラウンドで開始しました。"
echo "進捗確認は 'tail -f $LOG_FILE' で行ってください。"
echo "PID: $!" # プロセスIDを表示

echo "Setup and Training Script Finished."