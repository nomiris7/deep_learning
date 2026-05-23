import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold
import trimesh
import kagglehub

# Lista klas ModelNet10
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


# Pobieranie danych via kagglehub (kompletne – lokalna adaptacja)
def download_modelnet10():
    path = kagglehub.dataset_download("balraj98/modelnet10-princeton-3d-object-dataset")
    print("Ścieżka do plików datasetu (pobrane przez kagglehub):", path)

    data_dir = os.path.join(path, "ModelNet10")
    if not os.path.exists(data_dir):
        data_dir = path
        print(f"Brak podkatalogu ModelNet10, używam bezpośredniego path: {data_dir}")

    class_dir_example = os.path.join(data_dir, classes[0])  # np. bathtub
    train_dir_example = os.path.join(class_dir_example, "train")
    test_dir_example = os.path.join(class_dir_example, "test")

    if os.path.exists(train_dir_example) and os.path.exists(test_dir_example):
        print("Struktura potwierdzona: ModelNet10/<class_name>/train/ i test/")
    else:
        print("Błąd: Brak katalogów train/ lub test/ w", class_dir_example)
        print("Zawartość katalogu datasetu:", os.listdir(data_dir))
        print(
            "\nZawartość przykładowego katalogu klasy (bathtub):",
            os.listdir(class_dir_example),
        )
        raise FileNotFoundError(
            "Nie znaleziono oczekiwanej struktury <class_name>/train/test"
        )

    return data_dir


# Preprocessing: Samplowanie punktów z plików .off i zapis do HDF5
def preprocess_modelnet10(num_points=1024):
    """Przetwarza pliki .off do chmur punktów i zapisuje do lokalnego HDF5."""
    h5_path = "./data/modelnet10.h5"
    if os.path.exists(h5_path):
        print("Preprocessed dane już istnieją lokalnie w", h5_path)
        return h5_path

    data_dir = download_modelnet10()
    os.makedirs("./data", exist_ok=True)

    X_train, y_train = [], []
    X_test, y_test = [], []

    for split in ["train", "test"]:
        X_split, y_split = (X_train, y_train) if split == "train" else (X_test, y_test)
        print(f"Przetwarzanie zestawu {split}...")

        for class_name in classes:
            class_dir = os.path.join(data_dir, class_name, split)
            if not os.path.exists(class_dir):
                print(
                    f"Uwaga: Katalog {class_dir} nie istnieje, pomijam klasę {class_name} w {split}"
                )
                continue

            off_files = [f for f in os.listdir(class_dir) if f.endswith(".off")]
            print(f"  Klasa {class_name} ({split}): {len(off_files)} plików .off")

            for file in off_files:
                file_path = os.path.join(class_dir, file)
                try:
                    mesh = trimesh.load(file_path)
                    points, _ = trimesh.sample.sample_surface(mesh, num_points)
                    points = points - np.mean(points, axis=0)  # Centrowanie
                    points /= (
                        np.max(np.linalg.norm(points, axis=1)) + 1e-6
                    )  # Skalowanie
                    X_split.append(points)
                    y_split.append(class_to_idx[class_name])
                except Exception as e:
                    print(f"    Błąd przy pliku {file_path}: {e}")

        if split == "train":
            X_train = np.array(X_train)
            y_train = np.array(y_split)
        else:
            X_test = np.array(X_split)
            y_test = np.array(y_split)

    # Zapisz do HDF5 w lokalnym katalogu ./data
    with h5py.File(h5_path, "w") as hdf:
        hdf.create_dataset("X_train", data=X_train)
        hdf.create_dataset("y_train", data=y_train)
        hdf.create_dataset("X_test", data=X_test)
        hdf.create_dataset("y_test", data=y_test)

    print(
        f"Preprocessing zakończony. Kształty: Train {X_train.shape}, Test {X_test.shape}"
    )
    return h5_path


# Przygotowanie danych
class ModelNet10_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_data():
    h5_path = preprocess_modelnet10()

    with h5py.File(h5_path, "r") as hdf:
        X_train = hdf["X_train"][:]
        y_train = hdf["y_train"][:]
        X_test = hdf["X_test"][:]
        y_test = hdf["y_test"][:]

    val_size = 0.2
    num_train = len(X_train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(val_size * num_train))
    train_idx, val_idx = indices[split:], indices[:split]

    trainset = ModelNet10_Dataset(X_train[train_idx], y_train[train_idx])
    valset = ModelNet10_Dataset(X_train[val_idx], y_train[val_idx])
    testset = ModelNet10_Dataset(X_test, y_test)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    valloader = DataLoader(valset, batch_size=32, shuffle=False)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    print(
        f"Dane załadowane: Train {len(trainset)}, Val {len(valset)}, Test {len(testset)}"
    )
    return trainloader, valloader, testloader


# Definicja modelu PointNet
class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            torch.eye(self.k, dtype=torch.float32, device=x.device)
            .view(1, self.k * self.k)
            .repeat(batch_size, 1)
        )
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNet(nn.Module):
    def __init__(self, num_classes=10, num_points=1024):
        super(PointNet, self).__init__()
        self.num_classes = num_classes
        self.num_points = num_points

        self.tnet1 = TNet(k=3)
        self.tnet2 = TNet(k=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.transpose(2, 1)

        trans1 = self.tnet1(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans1)
        x = x.transpose(2, 1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        trans2 = self.tnet2(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans2)
        x = x.transpose(2, 1)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.bn6(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return self.logsoftmax(x)


# Funkcja trenująca
def train_model(
    model, trainloader, valloader, criterion, optimizer, num_epochs, patience=3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for points, labels in trainloader:
            points, labels = points.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * points.size(0)

        train_loss /= len(trainloader.sampler)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for points, labels in valloader:
                points, labels = points.to(device), labels.to(device)

                outputs = model(points)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * points.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(valloader.sampler)
        val_accuracy = 100 * correct / total

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Val Accuracy: {val_accuracy:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping przy epoce {epoch + 1}")
                model.load_state_dict(best_model)
                break

    return train_losses, val_losses, val_accuracies


# Funkcje wizualizacyjne
def plot_metrics(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.title("Strata Treningowa i Walidacyjna")
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Val Accuracy")
    plt.title("Dokładność Walidacji")
    plt.xlabel("Epoka")
    plt.ylabel("Dokładność (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_predictions(model, testloader, num_samples=5):
    device = next(model.parameters()).device
    model.eval()
    points, labels = next(iter(testloader))
    points, labels = points[:num_samples].to(device), labels[:num_samples].to(device)

    with torch.no_grad():
        outputs = model(points)
        _, predicted = torch.max(outputs, 1)

    points = points.cpu().numpy()

    fig = plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        ax = fig.add_subplot(1, num_samples, i + 1, projection="3d")
        ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2], s=1)
        ax.set_title(
            f"Pred: {classes[predicted[i].item()]}\nTrue: {classes[labels[i].item()]}"
        )
        ax.axis("off")
    plt.show()


# Walidacja krzyżowa
def cross_validation(k_folds=3, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h5_path = "./data/modelnet10.h5"

    with h5py.File(h5_path, "r") as hdf:
        full_data = hdf["X_train"][:]
        full_labels = hdf["y_train"][:]

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_data)):
        print(f"\n--- FOLD {fold + 1} / {k_folds} ---")

        fold_trainset = ModelNet10_Dataset(full_data[train_idx], full_labels[train_idx])
        fold_valset = ModelNet10_Dataset(full_data[val_idx], full_labels[val_idx])

        fold_trainloader = DataLoader(fold_trainset, batch_size=32, shuffle=True)
        fold_valloader = DataLoader(fold_valset, batch_size=32, shuffle=False)

        fold_model = PointNet(num_classes=10).to(device)
        fold_criterion = nn.CrossEntropyLoss()
        fold_optimizer = optim.Adam(fold_model.parameters(), lr=0.001)

        _, _, fold_val_accuracies = train_model(
            fold_model,
            fold_trainloader,
            fold_valloader,
            fold_criterion,
            fold_optimizer,
            num_epochs=num_epochs,
            patience=2,
        )

        best_acc = max(fold_val_accuracies) if fold_val_accuracies else 0.0
        results.append(best_acc)
        print(f"Fold {fold + 1} Najlepsza dokładność: {best_acc:.2f}%")

    print("\n========================================")
    print(
        f"Średnia Dokładność Walidacji Krzyżowej: {np.mean(results):.2f}% ± {np.std(results):.2f}%"
    )


# Główny blok wykonawczy – kluczowy przy uruchamianiu skryptów lokalnie
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uruchomiono model na urządzeniu: {device}")

    # Załadowanie i ewentualny preprocessing danych
    trainloader, valloader, testloader = load_data()

    model = PointNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Główny trening
    print("\nRozpoczynam trening główny...")
    train_losses, val_losses, val_accuracies = train_model(
        model, trainloader, valloader, criterion, optimizer, num_epochs=20, patience=5
    )

    # Wykresy i wizualizacje (okienka matplotlib otworzą się na pulpicie)
    plot_metrics(train_losses, val_losses, val_accuracies)
    visualize_predictions(model, testloader)

    # Walidacja krzyżowa
    print("\nRozpoczynam walidację krzyżową...")
    cross_validation(k_folds=3, num_epochs=5)

    # Ostateczny test na zbiorze testowym
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for points, labels in testloader:
            points, labels = points.to(device), labels.to(device)
            outputs = model(points)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"\nKońcowy wynik na zbiorze testowym (Test Accuracy): {100 * correct / total:.2f}%"
    )
