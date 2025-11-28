import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

# --- 設定パラメータ ---
IMG_SIZE = 224
BATCH_SIZE = 32
# 評価したいデータのパス
TEST_DATA_DIR = '../data/human' 
# 評価に使用するモデルのパス
MODEL_PATH = '../saved_models/efficientnetb0_fold1.h5'
# グラフ保存先
LOG_DIR = '../logs'

# --- 1. CLAHE 前処理 ---
def apply_clahe(image):
    image = image.astype('uint8')
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final.astype('float32')

def main():
    # ディレクトリ作成
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # 1. モデルのロード
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルファイルが見つかりません: {MODEL_PATH}")
        return

    print(f"モデルをロード中...: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # 2. テストデータジェネレータ
    test_datagen = ImageDataGenerator(
        preprocessing_function=apply_clahe,
        rescale=1./255
    )

    print("データを読み込んでいます...")
    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False  # 重要: 評価時はシャッフルしない
    )

    # 3. 推論の実行
    print("推論を実行中...")
    predictions = model.predict(test_generator, verbose=1)
    
    # 確率値 (0.0~1.0) を保持 (ROC/AUC用)
    y_score = predictions.ravel()
    
    # 0 or 1 のクラスに変換 (混同行列用)
    y_pred = (y_score > 0.5).astype(int)
    
    # 正解ラベル
    y_true = test_generator.classes
    
    # クラス名
    class_names = list(test_generator.class_indices.keys())

    # 4. 基本的な評価指標
    print("\n" + "="*30)
    print("       評価結果レポート")
    print("="*30)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    print(f"Accuracy (正解率): {accuracy:.4f}")
    print(f"Sensitivity (感度): {sensitivity:.4f}")
    print(f"Specificity (特異度): {specificity:.4f}")
    print(f"Precision (適合率): {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # --- 5. ROC曲線とAUCの計算・描画 ---
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    print(f"AUC Score: {roc_auc:.4f}")
    print("-" * 30)

    # ROC曲線のプロット
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(LOG_DIR, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC曲線を保存しました: {roc_path}")

    # 混同行列のプロット
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(LOG_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"混同行列を保存しました: {cm_path}")

if __name__ == '__main__':
    main()d