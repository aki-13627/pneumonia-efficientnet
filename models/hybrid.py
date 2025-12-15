import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
import cv2

from utils.smote import load_and_resample_with_smote

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
NUM_FOLDS = 5
EPOCHS = 30
IMG_SIZE = 128

DATA_DIR = './data/human'
SAVE_DIR = './saved_models'
LOG_DIR = './logs'

train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=[0.9, 1.1],
        brightness_range=[0.8, 1.2]
    )

val_datagen = ImageDataGenerator(
)

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def extract_patient_id(filename):
    name_body = os.path.splitext(filename)[0]
    
    if name_body.startswith("NORMAL2"):
        parts = name_body.split("-")
        if len(parts) >= 3:
            return f"patient_{parts[2]}"
            
    elif name_body.startswith("IM"):
        parts = name_body.split("-")
        if len(parts) >= 2:
            return f"patient_{parts[1]}"
            
    if "person" in name_body:
        return name_body.split("_")[0] 

    return name_body

class HybridCNNGRUSNN(nn.Module):
    def __init__(self, num_classes=2, time_steps=25, beta=0.95):
        super(HybridCNNGRUSNN, self).__init__()
        self.time_steps = time_steps
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
        )
        
        self.flatten = nn.Flatten()
        self.projection = nn.Sequential(
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        spike_grad = surrogate.fast_sigmoid()
        self.lif1_projection = nn.Linear(512, 256)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, output=True)

        self.fc1 = nn.Linear(768, 512)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.projection(x)
        
        x_temporal = x.unsqueeze(1).repeat(1, self.time_steps, 1)
        
        gru_out, _ = self.gru(x_temporal)
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        fusion_rec = []
        
        for step in range(self.time_steps):
            current_gru = gru_out[:, step, :]
            lif1_in = self.lif1_projection(current_gru)
            spk1, mem1 = self.lif1(lif1_in, mem1)
            fusion = torch.cat((current_gru, spk1), dim=1)
            fusion_rec.append(fusion)
            
            _, mem2 = self.lif2(fusion, mem2)

        fusion_stack = torch.stack(fusion_rec, dim=1)
        final_features = torch.mean(fusion_stack, dim=1)

        x = self.fc1(final_features)
        x = self.dropout1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        
        return x
    
def save_learning_curves(history, fold_no):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title(f'Fold {fold_no} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title(f'Fold {fold_no} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, f'learning_curve_fold{fold_no}.png'))
    plt.close()


def train_model(model, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}, Val Loss: {val_loss/len(val_loader)}, Accuracy: {100*correct/total}%")

def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100*correct/total}%")

def main():
    filepaths = []
    labels = []
    if os.path.exists(DATA_DIR):
        classes = sorted(os.listdir(DATA_DIR))
        for class_name in classes:
            class_dir = os.path.join(DATA_DIR, class_name)
            if not os.path.isdir(class_dir): continue
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepaths.append(os.path.join(class_dir, f))
                    labels.append(class_name)
    
    df = pd.DataFrame({'path': filepaths, 'class': labels})
    print(f"Total images: {len(df)}")
    
    df['filename'] = df['path'].apply(lambda x: os.path.basename(x))
    df['patient_id'] = df['filename'].apply(extract_patient_id)
    df['class_code'] = df['class'].astype('category').cat.codes # 0: Normal, 1: Pneumonia
    
    sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_no = 1
    
    for train_idx, val_idx in sgkf.split(df, df['class'], groups=df['patient_id']):
        print(f"\n{'='*20} Training Fold {fold_no} {'='*20}")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        X_train, y_train = load_and_resample_with_smote(train_df, IMG_SIZE)

        X_val = []
        y_val = []
        for _, row in val_df.iterrows():
            img = cv2.imread(row['path'])
            if img is None: continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            X_val.append(img)
            y_val.append(row['class_code'])
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
        val_gen = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
        
       
        steps_per_epoch = len(X_train) // BATCH_SIZE
        validation_steps = len(X_val) // BATCH_SIZE
        
        model = HybridCNNGRUSNN(num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=1e-4, 
            steps_per_epoch=steps_per_epoch, 
            epochs=EPOCHS
        )
        
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        
        
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for step in range(steps_per_epoch):
                inputs, targets = next(train_gen)
                inputs = torch.tensor(inputs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                targets = torch.tensor(targets, dtype=torch.long).to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step() 
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            
            epoch_loss = running_loss / steps_per_epoch
            epoch_acc = 100 * correct / total
            
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for step in range(validation_steps):
                    inputs, targets = next(val_gen)
                    inputs = torch.tensor(inputs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                    targets = torch.tensor(targets, dtype=torch.long).to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            val_epoch_loss = val_running_loss / validation_steps
            val_epoch_acc = 100 * val_correct / val_total
            
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            history['val_loss'].append(val_epoch_loss)
            history['val_acc'].append(val_epoch_acc)
            
            print(f"Epoch [{epoch+1}/{EPOCHS}] "
                  f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | "
                  f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.2f}%")
            
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_model_weights = model.state_dict().copy() 
                
        if best_model_weights is not None:
            save_path = os.path.join(SAVE_DIR, f'best_model_fold{fold_no}.pth')
            torch.save(best_model_weights, save_path)
            print(f"\nFold {fold_no} completed. Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
        else:
            print(f"\nFold {fold_no} completed. No model weights were saved.")

        df_log = pd.DataFrame(history)
        df_log.to_csv(os.path.join(LOG_DIR, f'training_log_fold{fold_no}.csv'), index=False)
        
        save_learning_curves(history, fold_no)
        
        fold_no += 1
        
    print("All folds training completed.")

if __name__ == "__main__":
    main()