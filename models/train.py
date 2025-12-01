import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)


IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_HEAD = 40
EPOCHS_FINE = 10
NUM_FOLDS = 5
DATA_DIR = './data/human'
SAVE_DIR = './saved_models'
LOG_DIR = './logs'

FROZEN_LAYERS_COUNT = 100

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


def build_base_model():

    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x) 
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def plot_history(history1, history2, fold_no):
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    
    plt.axvline(x=EPOCHS_HEAD, color='gray', linestyle='--', label='Fine Tuning Start')
    plt.title(f'Fold {fold_no}: Accuracy (MobileNetV2)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(x=EPOCHS_HEAD, color='gray', linestyle='--', label='Fine Tuning Start')
    plt.title(f'Fold {fold_no}: Loss (MobileNetV2)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    save_path = os.path.join(LOG_DIR, f'fold{fold_no}_history.png')
    plt.savefig(save_path)
    plt.close()

def plot_all_folds_comparison(acc_list, loss_list):
    """全Foldの比較グラフ"""
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for i, acc in enumerate(acc_list):
        plt.plot(range(1, len(acc) + 1), acc, label=f'Fold {i+1}')
    plt.title('Training Accuracy History (MobileNetV2)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    for i, loss in enumerate(loss_list):
        plt.plot(range(1, len(loss) + 1), loss, label=f'Fold {i+1}')
    plt.title('Training Loss History (MobileNetV2)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(os.path.join(LOG_DIR, 'all_folds_comparison.png'))
    plt.close()

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

    print(f"\n--- Fold {fold_no} Evaluation Results (MobileNetV2) ---")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"AUC        : {roc_auc:.4f}")
    print("-" * 30)

    # ROC
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
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
    all_acc_histories = []
    all_loss_histories = []
    
    for train_index, val_index in kfold.split(df['filename'], df['class_code']):
        print(f"\n{'='*20} Fold {fold_no} / {NUM_FOLDS} (MobileNetV2) {'='*20}")
        
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]
        
        train_generator = train_datagen.flow_from_dataframe(
            train_df, x_col='filename', y_col='class',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE, class_mode='binary', shuffle=True
        )
        
        val_generator = val_datagen.flow_from_dataframe(
            val_df, x_col='filename', y_col='class',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
        )
        
        # 1. モデル構築
        model, base_model = build_base_model()

        # === STEP 1: ヘッドのみ学習 (転移学習) ===
        print(f"Step 1: Training Head (Frozen Body) for {EPOCHS_HEAD} epochs...")
        base_model.trainable = False
        
        model.compile(optimizer=Adam(learning_rate=1e-3),
                      loss='binary_crossentropy', metrics=['accuracy'])
        
        history1 = model.fit(train_generator, epochs=EPOCHS_HEAD, validation_data=val_generator)

        print(f"Step 2: Fine Tuning (Unfrozen Body) for {EPOCHS_FINE} epochs...")
        base_model.trainable = True
        for i, layer in enumerate(base_model.layers):
            if i < FROZEN_LAYERS_COUNT or isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
                
        model.compile(optimizer=Adam(learning_rate=1e-5),
                      loss='binary_crossentropy', metrics=['accuracy'])
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1
        )
        
        history2 = model.fit(
            train_generator, 
            epochs=EPOCHS_FINE, 
            validation_data=val_generator,
            callbacks=[reduce_lr]
        )
        
        plot_history(history1, history2, fold_no)
        all_acc_histories.append(history1.history['accuracy'] + history2.history['accuracy'])
        all_loss_histories.append(history1.history['loss'] + history2.history['loss'])

        evaluate_fold(model, val_generator, fold_no)
        
        model_path = os.path.join(SAVE_DIR, f'mobilenetv2_fold{fold_no}.keras')
        model.save(model_path)
        print(f"Saved model: {model_path}")
        
        fold_no += 1
        tf.keras.backend.clear_session()

    plot_all_folds_comparison(all_acc_histories, all_loss_histories)
    print("All training completed.")

if __name__ == '__main__':
    main()