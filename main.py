import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Lista klas CIFAR-100
classes = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]

# 1. Przygotowanie danych
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform
)
testset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform
)

# Podział na zestaw treningowy i walidacyjny
val_size = 0.2  # UZUPEŁNIONE: 20% danych na walidację zgodnie z opisem
num_train = len(trainset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(val_size * num_train))
train_idx, val_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

trainloader = DataLoader(trainset, batch_size=64, sampler=train_sampler)
valloader = DataLoader(trainset, batch_size=64, sampler=val_sampler)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


# 2. Definicja modelu
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # UZUPEŁNIONE: Definicja warstw konwolucyjnych i w pełni połączonych
        # Wejście: 3 kanały (RGB), obrazek 32x32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)  # Zmniejsza wymiary przestrzenne o połowę
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Po dwóch operacjach pooling rozmiar obrazka zmienia się z 32x32 -> 16x16 -> 8x8.
        # Ostatnia warstwa ma 128 kanałów, stąd wejście do FC: 128 * 8 * 8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 100)  # 100 klas wyjściowych

    def forward(self, x):
        # UZUPEŁNIONE: Przepływ danych przez sieć
        x = self.pool(
            self.relu(self.bn1(self.conv1(x)))
        )  # Krok 1: Conv -> BN -> ReLU -> Pool (Wynik: 64x16x16)
        x = self.pool(
            self.relu(self.bn2(self.conv2(x)))
        )  # Krok 2: Conv -> BN -> ReLU -> Pool (Wynik: 128x8x8)

        x = x.view(-1, 128 * 8 * 8)  # Spłaszczenie tensora (Flatten)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 3. Funkcja trenująca
def train_model(
    model, trainloader, valloader, criterion, optimizer, num_epochs, patience=3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model = None

    for epoch in range(num_epochs):
        # Pętla treningowa
        model.train()
        train_loss = 0.0
        # UZUPEŁNIONE: Implementacja pętli treningowej
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Czyszczenie gradientów
            outputs = model(images)  # Przebieg w przód
            loss = criterion(outputs, labels)  # Obliczanie straty
            loss.backward()  # Wsteczna propagacja
            optimizer.step()  # Aktualizacja wag

            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(trainloader.sampler)
        train_losses.append(train_loss)

        # Pętla walidacyjna
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        # UZUPEŁNIONE: Implementacja pętli walidacyjnej przy użyciu torch.no_grad()
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(valloader.sampler)
        val_accuracy = 100 * correct / total

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Val Accuracy: {val_accuracy:.2f}%"
        )

        # Wczesne zatrzymanie
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                model.load_state_dict(best_model)
                break

    return train_losses, val_losses, val_accuracies


# 4. Funkcje wizualizacyjne
def plot_metrics(train_losses, val_losses, val_accuracies):
    # UZUPEŁNIONE: Generowanie czytelnych wykresów z etykietami i legendą
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))

    # Wykres 1: Strata (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "b-o", label="Strata treningowa")
    plt.plot(epochs, val_losses, "r-o", label="Strata walidacyjna")
    plt.title("Funkcja straty w zależności od epoki")
    plt.xlabel("Epoka")
    plt.ylabel("Wartość straty")
    plt.grid(True)
    plt.legend()

    # Wykres 2: Dokładność (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, "g-o", label="Dokładność walidacyjna")
    plt.title("Dokładność walidacyjna w zależności od epoki")
    plt.xlabel("Epoka")
    plt.ylabel("Dokładność (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def visualize_predictions(model, testloader, num_images=5):
    model.eval()
    images, labels = next(iter(testloader))
    images, labels = images[:num_images].to(device), labels[:num_images].to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    images = images.cpu().numpy()
    images = images * 0.5 + 0.5  # Denormalizacja
    images = np.transpose(images, (0, 2, 3, 1))

    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.title(
            f"Pred: {classes[predicted[i].item()]}\nTrue: {classes[labels[i].item()]}"
        )
        plt.axis("off")
    plt.show()


# 5. Walidacja krzyżowa
def cross_validation(k_folds=5, num_epochs=5):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = []

    # Potrzebujemy indeksów całego zbioru treningowego
    indices = np.arange(len(trainset))

    # UZUPEŁNIONE: Pętla walidacji krzyżowej
    for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):
        print(f"\n--- FOLD {fold + 1} / {k_folds} ---")

        # Samplery dedykowane dla danego folda
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        # Nowe DataLoadery na podstawie podziału KFold
        fold_trainloader = DataLoader(trainset, batch_size=64, sampler=train_subsampler)
        fold_valloader = DataLoader(trainset, batch_size=64, sampler=val_subsampler)

        # Inicjalizacja nowego modelu, kryterium i optymalizatora na każdy fold
        fold_model = SimpleCNN()
        fold_criterion = nn.CrossEntropyLoss()
        fold_optimizer = optim.Adam(fold_model.parameters(), lr=0.001)

        # Trening (ustawiamy mniejsze patience na potrzeby CV)
        _, _, fold_val_accuracies = train_model(
            fold_model,
            fold_trainloader,
            fold_valloader,
            fold_criterion,
            fold_optimizer,
            num_epochs=num_epochs,
            patience=2,
        )

        # Wyciągamy najlepszy wynik dokładności uzyskany w tym foldzie
        best_acc = max(fold_val_accuracies)
        results.append(best_acc)
        print(f"Fold {fold + 1} Best Val Accuracy: {best_acc:.2f}%")

    print("\n================================")
    print(
        f"Average Validation Accuracy: {np.mean(results):.2f}% ± {np.std(results):.2f}%"
    )


# 6. Główny kod
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Uruchomiono na urządzeniu: {device}")

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trening z wczesnym zatrzymaniem
print("Rozpoczynam trening główny...")
train_losses, val_losses, val_accuracies = train_model(
    model, trainloader, valloader, criterion, optimizer, num_epochs=10, patience=3
)

# Wizualizacja wyników
plot_metrics(train_losses, val_losses, val_accuracies)
visualize_predictions(model, testloader)

# Opcjonalna walidacja krzyżowa (wykonaj, jeśli czas i zasoby na to pozwalają)
print("\nRozpoczynam walidację krzyżową...")
cross_validation(k_folds=5, num_epochs=5)

# 7. Końcowa ewaluacja na zestawie testowym
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nFinal Test Accuracy: {100 * correct / total:.2f}%")
