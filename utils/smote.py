from imblearn.over_sampling import SMOTE
import cv2
import numpy as np

def load_and_resample_with_smote(df, img_size):
    
    X = []
    y = []
    
    # 1. 画像をメモリに読み込む
    for idx, row in df.iterrows():
        # 画像の読み込み
        img_path = row['path']
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        X.append(img)
        y.append(row['class_code'])

    X = np.array(X)
    y = np.array(y)
    
    print(f"Original shape: {X.shape}, Class distribution: {np.bincount(y)}")
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)
    
    
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled_flat, y_resampled = smote.fit_resample(X_flat, y)
    
    X_resampled = X_resampled_flat.reshape(-1, img_size, img_size, 3)
    
    print(f"Resampled shape: {X_resampled.shape}, Class distribution: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled