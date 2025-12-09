import tensorflow as tf
import numpy as np
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 設定 =================
MODEL_PATH = './saved_models/mobilenetv2_fold2.keras'
TEST_DIR = './data/test'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# =======================================


def apply_clahe(image):
    image = image.astype('uint8')
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final.astype('float32')


def preprocess_wrapper(image, label):
    
    aug_img = tf.numpy_function(func=apply_clahe, inp=[image], Tout=tf.float32)
    
    
    aug_img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    
    
    aug_img = tf.keras.layers.Rescaling(1./255)(aug_img)
    
    return aug_img, label

def evaluate():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"エラー: モデルの読み込みに失敗しました。\n{e}")
        return

    print(f"Loading test data from {TEST_DIR}...")
    
    

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=None, 
        image_size=IMG_SIZE,
        shuffle=False
    )


    class_names = test_ds.class_names
    print(f"Class mapping: {class_names}")

    
    test_ds = test_ds.map(preprocess_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    # 3. バッチ化 (モデルに入力するため)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("Predicting...")
    predictions = model.predict(test_ds)
    
    if predictions.shape[1] == 1:
        y_pred = (predictions > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(predictions, axis=1)

    # 正解ラベルの取得 (バッチ化されたデータセットから結合して取得)
    y_true = np.concatenate([y for x, y in test_ds], axis=0)

    # ================= 評価結果 =================
    print("\n" + "="*30)
    print("       EVALUATION REPORT")
    print("="*30)

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\n{class_names[0]} (True) -> Pred {class_names[0]}: {cm[0][0]}")
    print(f"{class_names[0]} (True) -> Pred {class_names[1]}: {cm[0][1]} (誤検知)")
    print(f"{class_names[1]} (True) -> Pred {class_names[0]}: {cm[1][0]} (見逃し)")
    print(f"{class_names[1]} (True) -> Pred {class_names[1]}: {cm[1][1]}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("\n混同行列の画像を 'confusion_matrix.png' に保存しました。")

if __name__ == "__main__":
    evaluate()