import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class DynamicCNN(nn.Module):
    def __init__(self, arch_json, num_classes=10):
        super(DynamicCNN, self).__init__()
        self.features = nn.Sequential()

        in_channels = 3  # CIFAR-10 to obrazy RGB

        for i, layer_def in enumerate(arch_json.get("layers", [])):
            l_type = layer_def["type"].lower()

            if l_type == "conv":
                out_channels = layer_def.get("filters", 32)
                kernel_size = layer_def.get("kernel", 3)
                self.features.add_module(
                    f"conv_{i}",
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                )
                in_channels = out_channels
            elif l_type == "relu":
                self.features.add_module(f"relu_{i}", nn.ReLU())
            elif l_type == "maxpool":
                self.features.add_module(f"pool_{i}", nn.MaxPool2d(2, 2))
            elif l_type == "batchnorm":
                self.features.add_module(f"bn_{i}", nn.BatchNorm2d(in_channels))
            elif l_type == "dropout":
                self.features.add_module(
                    f"drop_{i}", nn.Dropout(layer_def.get("p", 0.5))
                )

        # Spłaszczenie do wektora przed warstwą klasyfikującą
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_and_evaluate_cnn(arch_json, epochs=5):
    """
    Buduje, trenuje i ocenia wygenerowaną architekturę na CIFAR-10.
    """
    try:
        # 1. Wykrywanie sprzętu (CUDA dla NVIDIA, MPS dla Apple Silicon, inaczej CPU)
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Rozpoczynam trening na: {device}")

        # 2. Transformacje i pobieranie danych (download=True załatwia sprawę)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        # Jeśli zbiór nie istnieje w './data', zostanie pobrany automatycznie
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        train_loader = DataLoader(
            train_dataset, batch_size=128, shuffle=True, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=128, shuffle=False, num_workers=2
        )

        # 3. Inicjalizacja modelu
        model = DynamicCNN(arch_json).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 4. Pętla treningowa
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(
                f"Epoka {epoch + 1}/{epochs} zakończona. Loss: {running_loss / len(train_loader):.4f}"
            )

        # 5. Ewaluacja - obliczanie metryki Accuracy na zbiorze testowym
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    except Exception as e:
        print(f"[PyTorch Error] Błąd podczas treningu: {e}")
        return 0.0
