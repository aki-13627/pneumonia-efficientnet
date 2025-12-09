import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_FOLDS = 5
DATA_DIR = './data/human'


EPOCHS_HEAD = 40
OPTIMAL_LR_HEAD = 0.006886023869360358
OPTIMAL_DROPOUT = 0.3796178577283182
OPTIMAL_FROZEN_LAYERS = 66


def set_gpu_config():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            pass

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
    rotation_range=40, shear_range=0.2, zoom_range=0.2, 
    horizontal_flip=True, brightness_range=[0.5, 1.5], fill_mode='nearest'
)


val_datagen = ImageDataGenerator(
    preprocessing_function=apply_clahe,
    rescale=1./255
)

def build_fixed_model():
    """
    Dropout率を固定したモデルを構築
    """
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    x = Dropout(OPTIMAL_DROPOUT)(x) 
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def objective(trial, train_df, val_df):
    set_gpu_config()
    
    
    
    epochs_fine = trial.suggest_int("epochs_fine", 10, 50)
    
    lr_fine = trial.suggest_float("lr_fine", 1e-6, 5e-4, log=True)
    
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df, x_col='filename', y_col='class',
        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, 
        class_mode='binary', shuffle=True
    )
    val_generator = val_datagen.flow_from_dataframe(
        val_df, x_col='filename', y_col='class',
        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, 
        class_mode='binary', shuffle=False
    )

    model, base_model = build_fixed_model()

    try:
        
        base_model.trainable = False
        model.compile(optimizer=Adam(learning_rate=OPTIMAL_LR_HEAD),
                      loss='binary_crossentropy', metrics=['accuracy'])
        
        
        model.fit(train_generator, epochs=EPOCHS_HEAD, verbose=0)
        
        
        base_model.trainable = True 
        
        
        for i, layer in enumerate(base_model.layers):
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            elif i < OPTIMAL_FROZEN_LAYERS:
                layer.trainable = False
            else:
                layer.trainable = True

        
        model.compile(optimizer=Adam(learning_rate=lr_fine),
                      loss='binary_crossentropy', metrics=['accuracy'])
        
        
        model.fit(
            train_generator, 
            epochs=epochs_fine, 
            validation_data=val_generator,
            verbose=0 
        )
        
        
        y_true = val_generator.classes
        predictions = model.predict(val_generator, verbose=0)
        auc_score = roc_auc_score(y_true, predictions.ravel())
        
        return auc_score
        
    except Exception as e:
        print(f"Trial failed: {e}")
        
        raise optuna.exceptions.TrialPruned()
        
    finally:
        tf.keras.backend.clear_session()

if __name__ == '__main__':
    
    filepaths = []
    labels = []
    for class_name in sorted(os.listdir(DATA_DIR)):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir): continue
        for f in os.listdir(class_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepaths.append(os.path.join(class_dir, f))
                labels.append(class_name)
    
    df = pd.DataFrame({'filename': filepaths, 'class': labels})
    df['class_code'] = df['class'].astype('category').cat.codes
    
    
    kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    train_index, val_index = next(kfold.split(df['filename'], df['class_code']))
    
    train_df = df.iloc[train_index]
    val_df = df.iloc[val_index]

    print(f"--- Optuna HPO Start (Optimizing Epochs & LR for Fine Tuning) ---")
    print(f"Fixed LR_HEAD: {OPTIMAL_LR_HEAD}")
    print(f"Fixed Dropout: {OPTIMAL_DROPOUT}")
    print(f"Fixed Frozen Layers: {OPTIMAL_FROZEN_LAYERS}")
    
    
    study = optuna.create_study(direction='maximize', study_name="FineTuning_Epochs_LR_Opt")
    study.optimize(lambda trial: objective(trial, train_df, val_df), n_trials=50, show_progress_bar=True)
    
    print("\n--- HPO Results ---")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best AUC: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")