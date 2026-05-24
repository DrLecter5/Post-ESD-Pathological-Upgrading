import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import zipfile
import re
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
import warnings

warnings.filterwarnings('ignore')

# 1. Config
MODEL_PATHS = [
    '/workspace/checkpoints/病理升级_全景图_单流CBAM_单折微调/fold1_single_fold_best_ema.pth',
    '/workspace/checkpoints/病理升级_全景图_单流CBAM_单折微调/fold2_single_fold_best_ema.pth',
    '/workspace/checkpoints/病理升级_全景图_单流CBAM_单折微调/fold3_single_fold_best_ema.pth',
    '/workspace/checkpoints/病理升级_全景图_单流CBAM_单折微调/fold4_single_fold_best_ema.pth',
    '/workspace/checkpoints/病理升级_全景图_单流CBAM_单折微调/fold5_single_fold_best_ema.pth'
]

ZIP_PATHS = {
    0: ['/workspace/低级别.zip', '/workspace/病理降级.zip'], 
    1: ['/workspace/病理升级1.zip', '/workspace/病理升级2.zip']  
}

DATA_ROOT = '/workspace/dataset_eval_all'
SAVE_CSV_PATH = '/workspace/Patient_CNN_Predictions_OOF.csv'

CONFIG = {
    'img_size': 256,
    'batch_size': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'topk': 2,
    'seed': 2024,
    'n_splits': 5
}

# 2. Model Definition
class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        hidden = max(in_planes // ratio, 8)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, 1, bias=False),
            nn.Sigmoid(),
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        out = x * self.ca(x)
        spatial_in = torch.cat([torch.mean(out, dim=1, keepdim=True), torch.max(out, dim=1, keepdim=True)[0]], dim=1)
        out = out * self.sa(spatial_in)
        return out

class SingleStreamCBAMModel(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b1", pretrained=False, num_classes=0, global_pool="")
        self.feature_dim = self.backbone.num_features
        self.cbam = CBAMBlock(self.feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feature_dim, 2)

    def forward(self, x):
        fmap = self.backbone.forward_features(x)
        fmap = self.cbam(fmap)
        feat = fmap.mean(dim=(2, 3))
        feat = self.dropout(feat)
        return self.fc(feat)

# 3. Data Processing
def normalize_patient_name(x: str) -> str:
    x = "" if pd.isna(x) else str(x)
    x = x.replace("（", "(").replace("）", ")")
    x = re.sub(r"\s+", "", x)
    x = x.replace("_", "").strip()
    return x

def unzip_files():
    if not os.path.exists(DATA_ROOT) or len(os.listdir(DATA_ROOT)) == 0:
        os.makedirs(DATA_ROOT, exist_ok=True)
        for label, zip_list in ZIP_PATHS.items():
            target_dir = os.path.join(DATA_ROOT, str(label))
            os.makedirs(target_dir, exist_ok=True)
            for zip_path in zip_list:
                if os.path.exists(zip_path):
                    with zipfile.ZipFile(zip_path, "r") as z:
                        for file_info in z.infolist():
                            orig_filename = file_info.filename
                            try: 
                                decoded_name = orig_filename.encode("cp437").decode("gbk")
                            except: 
                                decoded_name = orig_filename.encode("cp437").decode("utf-8", errors="ignore")
                            
                            target_path = os.path.join(target_dir, decoded_name)
                            if file_info.is_dir():
                                os.makedirs(target_path, exist_ok=True)
                                continue
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            with open(target_path, "wb") as f:
                                f.write(z.read(orig_filename))
    print("Data extraction complete.")

def scan_data():
    image_paths, labels, groups = [], [], []
    for class_name in ["0", "1"]:
        class_path = os.path.join(DATA_ROOT, class_name)
        if not os.path.exists(class_path): continue
        current_label = int(class_name)

        for root, dirs, files in os.walk(class_path):
            if "__MACOSX" in root: continue
            images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if not images: continue
            
            patient_id = normalize_patient_name(os.path.basename(root))
            for f in images:
                p = os.path.join(root, f)
                s = p.lower()
                if any(k in s for k in ["副本", "局部", "细节", "放大", "close", "detail"]): continue
                image_paths.append(p)
                labels.append(current_label)
                groups.append(patient_id)
                
    return np.array(image_paths), np.array(labels), np.array(groups)

class InferenceDataset(Dataset):
    def __init__(self, img_paths, labels, patients, transform):
        self.img_paths = img_paths
        self.labels = labels
        self.patients = patients
        self.transform = transform

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            img = np.zeros((CONFIG['img_size'], CONFIG['img_size'], 3), dtype=np.uint8)
        img_tensor = self.transform(img)
        return img_tensor, self.labels[idx], self.patients[idx]

# 4. OOF Inference Loop
def run_oof_inference():
    unzip_files()
    X, y, groups = scan_data()
    print(f"Total images: {len(X)} | Total patients: {len(np.unique(groups))}")

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    sgkf = StratifiedGroupKFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=CONFIG['seed'])
    
    all_oof_probs = []
    all_oof_labels = []
    all_oof_patients = []

    model = SingleStreamCBAMModel(dropout=0.0).to(CONFIG['device'])

    for fold, (t_idx, v_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        print(f"Processing Fold {fold+1}/{CONFIG['n_splits']}...")
        
        weight_path = MODEL_PATHS[fold]
        if not os.path.exists(weight_path):
            print(f"Warning: weights not found at {weight_path}")
            continue
            
        model.load_state_dict(torch.load(weight_path, map_location=CONFIG['device']), strict=True)
        model.eval()

        X_val, y_val, g_val = X[v_idx], y[v_idx], groups[v_idx]
        
        ds_val = InferenceDataset(X_val, y_val, g_val, val_transform)
        dl_val = DataLoader(ds_val, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

        with torch.no_grad():
            for imgs, batch_labels, batch_patients in dl_val:
                imgs = imgs.to(CONFIG['device'])
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)[:, 1] 
                
                all_oof_probs.extend(probs.cpu().numpy())
                all_oof_labels.extend(batch_labels.numpy())
                all_oof_patients.extend(batch_patients)

    print("Aggregating patient-level predictions...")
    df = pd.DataFrame({'PatientID': all_oof_patients, 'TrueLabel': all_oof_labels, 'Prob': all_oof_probs})
    
    rows = []
    for pid, g in df.groupby("PatientID"):
        arr = np.sort(g["Prob"].values)
        kk = min(CONFIG['topk'], len(arr))
        prob_topk = arr[-kk:].mean() 
        prob_max = float(np.max(arr))
        
        rows.append({
            "PatientID": pid,
            "TrueLabel": int(g["TrueLabel"].iloc[0]),
            "CNNpred_Mean": float(prob_topk),
            "CNNpred_Max": prob_max,
            "ImageCount": len(g)
        })

    patient_df = pd.DataFrame(rows).sort_values("CNNpred_Mean", ascending=False)
    patient_df.to_csv(SAVE_CSV_PATH, index=False, encoding='utf-8-sig')
    
    print(f"OOF predictions saved to: {SAVE_CSV_PATH}")

if __name__ == '__main__':
    run_oof_inference()
