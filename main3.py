import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# --------------------
# 1. Konfiguracja Eksperymentu
# --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
epochs = 15

# Hiperparametry do zmiany podczas eksperymentów:
LATENT_DIM = 64  # Eksperyment: zmień na 128
USE_DROPOUT = False  # Eksperyment: ustaw True, aby sprawdzić wpływ regularyzacji

# --------------------
# 2. Załadowanie i Podział Danych CIFAR-10
# --------------------
# transforms.ToTensor() automatycznie mapuje piksele z [0, 255] do zakresu [0, 1]
transform = transforms.Compose([transforms.ToTensor()])

full_train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

# Podział zbioru treningowego na Train i Validation (45000 / 5000)
train_size = 45000
val_size = 5000
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# --------------------
# 3. Architektura Konwolucyjnego Autoenkodera
# --------------------
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim, use_dropout=False):
        super().__init__()
        p = 0.2 if use_dropout else 0.0

        # Encoder: co najmniej 2 warstwy konwolucyjne
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(
                3, 16, kernel_size=3, stride=2, padding=1
            ),  # wejście: 3x32x32 -> wyjście: 16x16x16
            nn.ReLU(),
            nn.Dropout2d(p),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # wyjście: 32x8x8
            nn.ReLU(),
            nn.Flatten(),
        )
        self.encoder_fc = nn.Linear(32 * 8 * 8, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 32 * 8 * 8)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=4, stride=2, padding=1
            ),  # wyjście: 16x16x16
            nn.ReLU(),
            nn.Dropout2d(p),
            nn.ConvTranspose2d(
                16, 3, kernel_size=4, stride=2, padding=1
            ),  # wyjście: 3x32x32
            nn.Sigmoid(),  # Gwarantuje wartości wyjściowe w przedziale [0, 1]
        )

    def forward(self, x):
        # Kodowanie
        x = self.encoder_conv(x)
        z = self.encoder_fc(x)
        # Dekodowanie
        x = self.decoder_fc(z)
        x = x.view(-1, 32, 8, 8)
        return self.decoder_conv(x)


model = ConvAutoencoder(LATENT_DIM, USE_DROPOUT).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --------------------
# 4. Trening i Walidacja
# --------------------
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    t_loss = 0
    for x_batch, _ in train_loader:
        x_batch = x_batch.to(device)
        recon = model(x_batch)
        loss = criterion(recon, x_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t_loss += loss.item() * x_batch.size(0)

    # Walidacja
    model.eval()
    v_loss = 0
    with torch.no_grad():
        for x_batch, _ in val_loader:
            x_batch = x_batch.to(device)
            recon = model(x_batch)
            loss = criterion(recon, x_batch)
            v_loss += loss.item() * x_batch.size(0)

    train_losses.append(t_loss / train_size)
    val_losses.append(v_loss / val_size)
    print(
        f"Epoch {epoch + 1:02d} | Train MSE: {train_losses[-1]:.5f} | Val MSE: {val_losses[-1]:.5f}"
    )

# --------------------
# 5. Ewaluacja i Wykresy Wyników
# --------------------
model.eval()
all_latents, all_labels = [], []
test_loss = 0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        # Wyciągamy wektor latentny bezpośrednio
        flat_conv = model.encoder_conv(x_batch)
        z = model.encoder_fc(flat_conv)
        recon = model(x_batch)

        test_loss += criterion(recon, x_batch).item() * x_batch.size(0)
        all_latents.append(z.cpu().numpy())
        all_labels.append(y_batch.numpy())

all_latents = np.concatenate(all_latents)
all_labels = np.concatenate(all_labels)
print(
    f"\nŚredni błąd rekonstrukcji (MSE) dla CIFAR-10: {test_loss / len(test_dataset):.5f}"
)

# Wizualizacja oryginalnych obrazów vs rekonstrukcje
images, _ = next(iter(test_loader))
with torch.no_grad():
    recon_images = model(images.to(device)).cpu()

n = 6
plt.figure(figsize=(12, 5))
for i in range(n):
    # Oryginały
    plt.subplot(2, n, i + 1)
    plt.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
    plt.title("Oryginał")
    plt.axis("off")

    # Rekonstrukcje
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(np.transpose(recon_images[i].numpy(), (1, 2, 0)))
    plt.title("Rekonstrukcja")
    plt.axis("off")
plt.suptitle("CIFAR-10: Porównanie rekonstrukcji")
plt.show()

# Redukcja przestrzeni latentnej CIFAR-10 do 2D przy użyciu PCA
pca_cifar = PCA(n_components=2)
z_2d = pca_cifar.fit_transform(all_latents)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    z_2d[:, 0], z_2d[:, 1], c=all_labels, cmap="tab10", s=5, alpha=0.6
)
plt.colorbar(scatter, ticks=range(10), label="Klasy CIFAR-10")
plt.title(
    f"Przestrzeń latentna CIFAR-10 zredukowana do 2D (PCA) | Latent dim = {LATENT_DIM}"
)
plt.show()
