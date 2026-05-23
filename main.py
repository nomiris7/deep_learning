import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import Food101
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIG
# ==========================================
BATCH_SIZE = 32
NUM_CLASSES = 101
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 5
EARLY_STOPPING_PATIENCE = 2

# ==========================================
# 2. DEFINICJA TRANSFORMACJI I ŁADOWANIE DATASETU
# ==========================================
train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Pobieranie pełnego datasetu bazowego
full_train_dataset = Food101(
    root="./data", split="train", download=True, transform=train_transform
)
test_dataset = Food101(
    root="./data", split="test", download=True, transform=test_transform
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Wyciągamy etykiety do podziału stratyfikowanego
# Uwaga: w Food101 etykiety są w full_train_dataset._labels
targets = np.array(full_train_dataset._labels)
indices = np.arange(len(full_train_dataset))


# --- FUNKCJA TWORZĄCA PODZIAŁY 1% ORAZ 10% ---
def get_stratified_loaders(train_ratio):
    """Tworzy stratyfikowany podział i zwraca DataLoader dla Train oraz Val"""
    # Wycinamy żądany % danych z całości
    train_idx, _, y_train, _ = train_test_split(
        indices, targets, train_size=train_ratio, stratify=targets, random_state=42
    )

    # Robimy podział na train (80%) i validation (20%) wewnątrz wyciętej próbki
    t_idx, v_idx, _, _ = train_test_split(
        train_idx, y_train, train_size=0.8, stratify=y_train, random_state=42
    )

    train_subset = Subset(full_train_dataset, t_idx)
    # Dla walidacji podmieniamy transformacje na testowe (bez augmentacji)
    val_dataset_copied = Food101(
        root="./data", split="train", download=False, transform=test_transform
    )
    val_subset = Subset(val_dataset_copied, v_idx)

    t_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    v_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    return t_loader, v_loader


# ==========================================
# 3. UNIWERSALNA FUNKCJA TRENINGOWA
# ==========================================
def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    epochs=5,
    name="Model",
    patience=2,
):
    train_losses, val_losses, val_accs = [], [], []
    best_loss = float("inf")
    no_improve = 0
    best_state = None

    model.to(DEVICE)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0
        correct, total = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        acc = correct / total if total > 0 else 0

        val_losses.append(val_loss)
        val_accs.append(acc)

        print(
            f"[{name}] Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses, val_accs


# ==========================================
# 4. FUNKCJE INICJALIZUJĄCE MODELE
# ==========================================
def get_efficientnet():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, NUM_CLASSES)
    return model


def get_convnext():
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
    return model


# Słownik do zapisu wyników końcowych
results = {}

# ==========================================
# 5. PĘTLA EKSPERYMENTU (1% oraz 10%)
# ==========================================
splits = {"1pct": 0.01, "10pct": 0.10}

for split_name, split_ratio in splits.items():
    print(f"\n==========================================")
    # Pobieramy dedykowane dla danego procentu podziały danych
    t_loader, v_loader = get_stratified_loaders(split_ratio)
    print(
        f"URUCHAMIANIE EKSPERYMENTU DLA PODZIAŁU: {split_name.upper()} ({split_ratio * 100}%)"
    )
    print(
        f"Liczba batchy treningowych: {len(t_loader)}, walidacyjnych: {len(v_loader)}"
    )
    print(f"==========================================")

    # --- Model 1: EfficientNet ---
    print(f"\n--- Trening EfficientNet na podziale {split_name} ---")
    eff_model = get_efficientnet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(eff_model.classifier[1].parameters(), lr=1e-3)

    _, _, eff_acc = train_model(
        eff_model,
        criterion,
        optimizer,
        t_loader,
        v_loader,
        epochs=EPOCHS,
        name=f"EfficientNet_{split_name}",
        patience=EARLY_STOPPING_PATIENCE,
    )
    results[f"EfficientNet_{split_name}"] = eff_acc

    # --- Model 2: ConvNeXt ---
    print(f"\n--- Trening ConvNeXt na podziale {split_name} ---")
    conv_model = get_convnext()
    optimizer = optim.Adam(conv_model.classifier[2].parameters(), lr=1e-3)

    _, _, conv_acc = train_model(
        conv_model,
        criterion,
        optimizer,
        t_loader,
        v_loader,
        epochs=EPOCHS,
        name=f"ConvNeXt_{split_name}",
        patience=EARLY_STOPPING_PATIENCE,
    )
    results[f"ConvNeXt_{split_name}"] = conv_acc

# ==========================================
# 6. GENEROWANIE WYKRESU PORÓWNAWCZEGO
# ==========================================
plt.figure(figsize=(10, 6))
for model_label, acc_history in results.items():
    plt.plot(acc_history, label=model_label, marker="o")

plt.title("Porównanie dokładności (Validation Accuracy) - Transfer Learning")
plt.xlabel("Epoka")
plt.ylabel("Dokładność")
plt.grid(True)
plt.legend()
plt.show()
