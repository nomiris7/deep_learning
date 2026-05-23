import os
import h5py
import numpy as np
import trimesh
import kagglehub
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

# ============================================
# CONFIG & SETTINGS
# ============================================
classes = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================
# POBIERANIE I PREPROCESSING (ZAPIS DO HDF5)
# ============================================
def download_modelnet10():
    path = kagglehub.dataset_download("balraj98/modelnet10-princeton-3d-object-dataset")
    data_dir = os.path.join(path, "ModelNet10")
    if not os.path.exists(data_dir):
        data_dir = path
    print("Dataset root:", data_dir)
    return data_dir


def preprocess_modelnet10(num_points=1024):
    h5_path = "modelnet10.h5"
    if os.path.exists(h5_path):
        print("HDF5 już istnieje:", h5_path)
        return h5_path

    data_dir = download_modelnet10()

    X_train, y_train = [], []
    X_test, y_test = [], []

    for split in ["train", "test"]:
        X_out = X_train if split == "train" else X_test
        y_out = y_train if split == "train" else y_test

        print(f"Przetwarzanie {split}...")
        for cls in classes:
            class_dir = os.path.join(data_dir, cls, split)
            if not os.path.exists(class_dir):
                continue
            off_files = [f for f in os.listdir(class_dir) if f.endswith(".off")]

            for off in off_files:
                path = os.path.join(class_dir, off)
                try:
                    mesh = trimesh.load(path)
                    pts, _ = trimesh.sample.sample_surface(mesh, num_points)
                    pts = pts - pts.mean(axis=0)
                    pts /= np.max(np.linalg.norm(pts, axis=1)) + 1e-6
                    X_out.append(pts)
                    y_out.append(class_to_idx[cls])
                except Exception as e:
                    pass

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X_train", data=X_train)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("X_test", data=X_test)
        f.create_dataset("y_test", data=y_test)

    print("Zapisano HDF5:", h5_path)
    return h5_path


class ModelNet10Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# --- STRATIFIED LOADER GENERATOR ---
def get_stratified_loaders(h5_path, train_ratio):
    """Tworzy stratyfikowany podział danych bez wywoływania błędu dla małych prób."""
    with h5py.File(h5_path, "r") as f:
        X_train_all = f["X_train"][:]
        y_train_all = f["y_train"][:]

    indices = np.arange(len(X_train_all))
    full_dataset = ModelNet10Dataset(X_train_all, y_train_all)

    # Dla pretrainingu (np. 50% danych) robimy normalny podział wewnętrzny Train/Val
    if train_ratio > 0.15:
        train_idx, val_idx = train_test_split(
            indices, train_size=train_ratio, stratify=y_train_all, random_state=42
        )
        # Podział wewnątrz odciętej części
        t_idx, v_idx = train_test_split(
            train_idx, train_size=0.8, stratify=y_train_all[train_idx], random_state=42
        )
        train_loader = DataLoader(
            Subset(full_dataset, t_idx), batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            Subset(full_dataset, v_idx), batch_size=BATCH_SIZE, shuffle=False
        )

    # Dla skrajnie małych wartości (1% oraz 10%)
    else:
        # Wycinamy dokładnie 1% lub 10% jako zbiór treningowy
        t_idx, val_fallback_idx = train_test_split(
            indices, train_size=train_ratio, stratify=y_train_all, random_state=42
        )

        # Jako zbiór walidacyjny bierzemy pozostałą, dużą część danych (reszta z podziału)
        # Zapobiega to crashowaniu i daje świetny, stabilny benchmark dla małego modelu
        _, v_idx = train_test_split(
            val_fallback_idx, train_size=0.8, test_size=0.2, random_state=42
        )

        train_loader = DataLoader(
            Subset(full_dataset, t_idx), batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            Subset(full_dataset, v_idx), batch_size=BATCH_SIZE, shuffle=False
        )

    return train_loader, val_loader


# ============================================
# ARCHITEKTURA POINTNET & TRANSFER LEARNING
# ============================================
class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x):
        B = x.size(0)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        I = torch.eye(self.k).to(x.device).view(1, self.k, self.k)
        return x.view(-1, self.k, self.k) + I


class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.t1 = TNet(3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.t2 = TNet(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        t = self.t1(x)
        x = torch.bmm(x.transpose(2, 1), t).transpose(2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        t2 = self.t2(x)
        x = torch.bmm(x.transpose(2, 1), t2).transpose(2, 1)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2)[0]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PointNet_FT(nn.Module):
    def __init__(self, pretrained_model, num_classes=10):
        super().__init__()
        self.backbone = pretrained_model

        # Podmieniamy klasyfikator końcowy na nowy
        self.backbone.fc1 = nn.Linear(1024, 512)
        self.backbone.fc2 = nn.Linear(512, 256)
        self.backbone.fc3 = nn.Linear(256, num_classes)

        # Zamrażamy wagi ekstraktora cech, odmrażamy tylko końcówkę (fc i conv3)
        for name, param in self.backbone.named_parameters():
            param.requires_grad = "conv3" in name or "fc" in name

    def forward(self, x):
        return self.backbone(x)


# ============================================
# UNIWERSALNA PĘTLA TRENINGOWA
# ============================================
def train_model(
    model, train_loader, val_loader, epochs, lr=1e-4, name="Model", patience=3
):
    model.to(DEVICE)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_weights = None
    no_improve = 0
    val_acc_history = []

    for epoch in range(epochs):
        model.train()
        for pts, lbl in train_loader:
            pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()
            out = model(pts)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()

        # Walidacja
        model.eval()
        correct, total, val_loss = 0, 0, 0
        with torch.no_grad():
            for pts, lbl in val_loader:
                pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
                out = model(pts)
                loss = criterion(out, lbl)
                val_loss += loss.item()
                _, pred = torch.max(out, 1)
                correct += (pred == lbl).sum().item()
                total += lbl.size(0)

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total if total > 0 else 0
        val_acc_history.append(val_acc)

        print(
            f"[{name}] Epoch {epoch + 1}/{epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping na epoce {epoch + 1}!")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model, val_acc_history


# ============================================
# GŁÓWNY PROCES EKSPERYMENTU
# ============================================
if __name__ == "__main__":
    h5_path = preprocess_modelnet10()

    # --- KROK 1: Przygotowanie "Pretrained" modelu bazowego ---
    # Symulujemy posiadanie bazowego modelu poprzez krótki trening na domyślnych danych
    print("\n--- Przygotowanie wag początkowych (Pretraining) ---")
    t_loader_init, v_loader_init = get_stratified_loaders(h5_path, train_ratio=0.5)
    base_model = PointNet(num_classes=10)
    base_model, _ = train_model(
        base_model, t_loader_init, v_loader_init, epochs=3, lr=1e-3, name="Pretraining"
    )
    torch.save(base_model.state_dict(), "pointnet_backbone.pth")

    # Słownik do przechowywania historii wyników dla wykresu
    experiment_results = {}

    # --- KROK 2: Testowanie podziałów transfer learningu (1% oraz 10%) ---
    splits = {"1% Danych": 0.01, "10% Danych": 0.10}

    for label, ratio in splits.items():
        print(f"\n==========================================")
        print(f"URUCHAMIANIE TRANSFER LEARNINGU DLA: {label}")
        print(f"==========================================")

        # Pobieramy dedykowany, stratyfikowany podział danych
        train_loader, val_loader = get_stratified_loaders(h5_path, train_ratio=ratio)

        # Świeża inicjalizacja struktury i załadowanie wag "wytrenowanych"
        fresh_backbone = PointNet(num_classes=10)
        fresh_backbone.load_state_dict(torch.load("pointnet_backbone.pth"))

        # Zamrożenie tyłu i przygotowanie głowicy klasyfikatora pod eksperyment
        ft_model = PointNet_FT(fresh_backbone, num_classes=10)

        # Trening właściwy
        _, acc_hist = train_model(
            ft_model,
            train_loader,
            val_loader,
            epochs=EPOCHS,
            lr=1e-4,
            name=f"FineTuning_{label.replace('%', '')}",
            patience=EARLY_STOPPING_PATIENCE,
        )

        experiment_results[label] = acc_hist

    # --- KROK 3: Generowanie wykresu porównawczego ---
    plt.figure(figsize=(9, 5))
    for run_name, history in experiment_results.items():
        plt.plot(history, marker="o", label=f"Transfer Learning ({run_name})")

    plt.title("Wpływ ilości danych na celność walidacji (ModelNet10)")
    plt.xlabel("Epoka")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
