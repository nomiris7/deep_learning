import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# --------------------
# 1. Konfiguracja Eksperymentu
# --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
n_samples = 5000
n_features = 77  # Możesz zmienić na 10, jeśli profesor wymaga dokładnie 10
n_clusters = 32

# Hiperparametry do zmiany podczas eksperymentów:
LATENT_DIM = 2  # Eksperyment: zmień na 5
USE_DROPOUT = False  # Eksperyment: zmień na True dla regularyzacji
WEIGHT_DECAY = 0.0  # Eksperyment: zmień na 1e-4 dla regularyzacji L2

# --------------------
# 2. Generowanie i Przygotowanie Danych
# --------------------
X, y = make_blobs(
    n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42
)

# Skalowanie cech do zakresu [0, 1] dla stabilności sieci
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Wizualizacja danych przed treningiem (PCA 2D)
pca_initial = PCA(n_components=2)
X_pca_initial = pca_initial.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_initial[:, 0], X_pca_initial[:, 1], c=y, cmap="tab20", s=10)
plt.title("Wizualizacja wejściowych danych syntetycznych (PCA 2D)")
plt.colorbar(label="Indeks klastra")
plt.show()

# Podział na Train, Val, Test (80% / 10% / 10%)
dataset = TensorDataset(
    torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
)
train_size = int(0.8 * n_samples)
val_size = int(0.1 * n_samples)
test_size = n_samples - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# --------------------
# 3. Model: Autoenkoder MLP
# --------------------
class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, use_dropout=False):
        super().__init__()
        p = 0.2 if use_dropout else 0.0

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),  # Ponieważ dane są wyskalowane do [0, 1]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


model = MLPAutoencoder(n_features, LATENT_DIM, USE_DROPOUT).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)

# --------------------
# 4. Pętla Treningowa z Walidacją
# --------------------
epochs = 30
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
    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1:02d} | Train MSE: {train_losses[-1]:.5f} | Val MSE: {val_losses[-1]:.5f}"
        )

# Wykres historii uczenia
plt.figure(figsize=(6, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
# Poprawne formatowanie osi i legendy za pomocą czystego markdown w tekście opisu
plt.title("Historia strat – Dane Syntetyczne")
plt.xlabel("Epoka")
plt.ylabel("MSE")
plt.legend()
plt.show()

# --------------------
# 5. Ewaluacja i Wizualizacja Przestrzeni Latentnej
# --------------------
model.eval()
all_latents, all_labels = [], []
test_loss = 0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        z = model.encoder(x_batch)
        recon = model(x_batch)
        test_loss += criterion(recon, x_batch).item() * x_batch.size(0)

        all_latents.append(z.cpu().numpy())
        all_labels.append(y_batch.numpy())

all_latents = np.concatenate(all_latents)
all_labels = np.concatenate(all_labels)
print(
    f"\nŚredni błąd rekonstrukcji (MSE) na zbiorze testowym: {test_loss / test_size:.5f}"
)

# Wizualizacja przestrzeni latentnej
plt.figure(figsize=(8, 6))
if LATENT_DIM == 2:
    plt.scatter(all_latents[:, 0], all_latents[:, 1], c=all_labels, cmap="tab20", s=15)
    plt.title("Przestrzeń Latentna 2D Autoenkodera (Kolory = Klastry)")
else:
    # Jeśli wymiar wynosi 5, musimy zredukować go do 2D za pomocą PCA, aby wyświetlić wykres
    pca_lat = PCA(n_components=2)
    lat_2d = pca_lat.fit_transform(all_latents)
    plt.scatter(lat_2d[:, 0], lat_2d[:, 1], c=all_labels, cmap="tab20", s=15)
    plt.title(f"Przestrzeń Latentna {LATENT_DIM}D zredukowana do 2D przez PCA")

plt.colorbar(label="Indeks klastra")
plt.show()

# Porównanie wejścia z wyjściem dla losowej próbki
sample_idx = 0
x_orig = test_dataset[sample_idx][0].numpy()
with torch.no_grad():
    x_recon = (
        model(test_dataset[sample_idx][0].unsqueeze(0).to(device)).cpu().numpy()[0]
    )

plt.figure(figsize=(10, 4))
plt.plot(x_orig, label="Oryginalny wektor (cechy)", alpha=0.7)
plt.plot(x_recon, label="Zrekonstruowany wektor", linestyle="--", alpha=0.7)
plt.title("Porównanie wartości wejściowych i wyjściowych (Próbka 0)")
plt.legend()
plt.show()
