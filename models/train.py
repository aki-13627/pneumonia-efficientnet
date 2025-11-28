import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Dropout # 追加
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)

# --- 設定パラメータ ---
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
NUM_FOLDS = 5
DATA_DIR = './data/human'
SAVE_DIR = './saved_models'
LOG_DIR = './logs'

def apply_clahe(image):
    image = image.astype('uint8')
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final.astype('float32')

train_datagen = ImageDataGenerator(
    preprocessing_function=apply_clahe,
    rescale=1./255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=apply_clahe,
    rescale=1./255
)

def build_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = True 
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_fold_history(history, fold_no):
    """単一Foldの学習曲線を保存"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title(f'Fold {fold_no}: Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title(f'Fold {fold_no}: Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(LOG_DIR, f'fold{fold_no}_history.png'))
    plt.close()

def plot_all_folds_comparison(histories):
    """全Foldの学習曲線を重ねて保存 (論文のFigure 11相当)"""
    plt.figure(figsize=(14, 6))


    plt.subplot(1, 2, 1)
    for i, history in enumerate(histories):
        acc = history.history['accuracy']
        plt.plot(range(1, len(acc) + 1), acc, label=f'Fold {i+1} Accuracy')
    
    plt.title('Training Accuracy History for Each Fold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)


    plt.subplot(1, 2, 2)
    for i, history in enumerate(histories):
        loss = history.history['loss']
        plt.plot(range(1, len(loss) + 1), loss, label=f'Fold {i+1} Loss')
    
    plt.title('Training Loss History for Each Fold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    save_path = os.path.join(LOG_DIR, 'all_folds_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"\n全Fold比較グラフを保存しました: {save_path}")

def evaluate_fold(model, val_generator, fold_no):
    y_true = val_generator.classes
    predictions = model.predict(val_generator, verbose=0)
    y_score = predictions.ravel()
    y_pred = (y_score > 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    print(f"\n--- Fold {fold_no} Evaluation Results ---")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print(f"AUC        : {roc_auc:.4f}")
    print("-" * 30)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pneumonia'], 
                yticklabels=['Normal', 'Pneumonia'])
    plt.title(f'Fold {fold_no} Confusion Matrix')
    plt.savefig(os.path.join(LOG_DIR, f'fold{fold_no}_confusion_matrix.png'))
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Fold {fold_no} ROC')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(LOG_DIR, f'fold{fold_no}_roc.png'))
    plt.close()

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    filepaths = []
    labels = []
    classes = sorted(os.listdir(DATA_DIR))
    
    print("Loading data...")
    for class_name in classes:
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = os.listdir(class_dir)
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepaths.append(os.path.join(class_dir, f))
                labels.append(class_name)
    
    df = pd.DataFrame({'filename': filepaths, 'class': labels})
    print(f"Total images: {len(df)}")
    
    kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    df['class_code'] = df['class'].astype('category').cat.codes
    
    fold_no = 1
    all_histories = []
    
    for train_index, val_index in kfold.split(df['filename'], df['class_code']):
        print(f"\n{'='*20} Fold {fold_no} / {NUM_FOLDS} {'='*20}")
        
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]
        
        print(f"Train: {len(train_df)}, Val: {len(val_df)}")
        
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='filename',
            y_col='class',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_dataframe(
            val_df,
            x_col='filename',
            y_col='class',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )
        
        model = build_model()
        
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator
        )
        
        # 履歴をリストに追加
        all_histories.append(history)
        
        plot_fold_history(history, fold_no)
        evaluate_fold(model, val_generator, fold_no)
        
        model_path = os.path.join(SAVE_DIR, f'efficientnetb0_fold{fold_no}.h5')
        model.save(model_path)
        print(f"Saved model: {model_path}")
        
        fold_no += 1
        tf.keras.backend.clear_session()

    plot_all_folds_comparison(all_histories)
    print("All training completed.")

if __name__ == '__main__':
    main()