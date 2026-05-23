import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# --------------------
# Configuration
# --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
epochs = 15  # GAN-y potrzebują nieco więcej czasu niż AE
latent_dim = 100  # Standardowy wymiar szumu wejściowego dla G
lr = 2e-4  # Optymalne hiperparametry dla stabilności DCGAN (Adam)
beta1 = 0.5

# WYBÓR ZBIORU DANYCH: Zmień na 'FashionMNIST' dla drugiego zbioru!
DATASET_TYPE = "MNIST"  # Opcje: "MNIST", "FashionMNIST"

# --------------------
# Data (Znormalizowane do zakresu [-1, 1] – kluczowe dla GANów)
# --------------------
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

if DATASET_TYPE == "MNIST":
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
elif DATASET_TYPE == "FashionMNIST":
    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)


# --------------------
# Model: Generator
# --------------------
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.main = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Stan: 128 x 7 x 7 -> Transponowany splot do 64 x 14 x 14
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Stan: 64 x 14 x 14 -> Transponowany splot do 1 x 28 x 28
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh(),  # Tanh mapuje wyjście do [-1, 1], dopasowując do danych wejściowych
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(-1, 128, 7, 7)
        return self.main(out)


# --------------------
# Model: Discriminator
# --------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # Wejście: 1 x 28 x 28 -> Splot do 64 x 14 x 14
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Stan: 64 x 14 x 14 -> Splot do 128 x 7 x 7
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),  # Zwraca prawdopodobieństwo: 1 = prawdziwy, 0 = sztuczny
        )

    def forward(self, x):
        return self.main(x)


# Inicjalizacja wag (rekomendowana dla DCGAN)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


netG = Generator(latent_dim).to(device)
netD = Discriminator().to(device)

netG.apply(weights_init)
netD.apply(weights_init)

criterion = (
    nn.BCELoss()
)  # Binary Cross Entropy idealnie nadaje się do klasyfikacji prawda/fałsz

# Osobne optimizery są absolutnie konieczne
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

# Stały szum do sprawdzania postępów generatora w trakcie epok
fixed_noise = torch.randn(64, latent_dim, device=device)

# --------------------
# Adversarial Training Loop
# --------------------
losses_G = []
losses_D = []
saved_images = []

print(f"Rozpoczynanie treningu GAN na zbiorze: {DATASET_TYPE}...")

for epoch in range(epochs):
    loss_G_epoch = 0
    loss_D_epoch = 0

    for i, (real_imgs, _) in enumerate(train_loader):
        current_batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        # Tworzenie etykiet (Soft labels mogą poprawić stabilność, tu klasyczne 1 i 0)
        label_real = torch.ones(current_batch_size, 1, device=device)
        label_fake = torch.zeros(current_batch_size, 1, device=device)

        # ---------------------
        # Krok 1: Trening Dyskryminatora (Max log(D(x)) + log(1 - D(G(z))))
        # ---------------------
        netD.zero_grad()

        # Prawdziwe obrazy
        output_real = netD(real_imgs)
        loss_D_real = criterion(output_real, label_real)

        # Sztuczne obrazy
        noise = torch.randn(current_batch_size, latent_dim, device=device)
        fake_imgs = netG(noise)
        output_fake = netD(
            fake_imgs.detach()
        )  # .detach(), żeby nie liczyć gradientów dla G
        loss_D_fake = criterion(output_fake, label_fake)

        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimizerD.step()

        # ---------------------
        # Krok 2: Trening Generatora (Max log(D(G(z))))
        # ---------------------
        netG.zero_grad()

        # Chcemy, żeby dyskryminator pomyślał, że sztuczne są prawdziwe (etykieta real)
        output_g = netD(fake_imgs)
        loss_G = criterion(output_g, label_real)

        loss_G.backward()
        optimizerG.step()

        # Statystyki
        loss_D_epoch += loss_D.item()
        loss_G_epoch += loss_G.item()

    # Średnie straty z epoki
    loss_D_epoch /= len(train_loader)
    loss_G_epoch /= len(train_loader)
    losses_D.append(loss_D_epoch)
    losses_G.append(loss_G_epoch)

    print(
        f"Epoch [{epoch + 1}/{epochs}] | Loss D: {loss_D_epoch:.4f}, Loss G: {loss_G_epoch:.4f}"
    )

    # Wizualizacja próbek generowanych na danym etapie treningu
    if (epoch + 1) % 3 == 0 or epoch == 0 or epoch == epochs - 1:
        netG.eval()
        with torch.no_grad():
            gen_imgs = netG(fixed_noise).detach().cpu()
            # Denormalizacja z [-1, 1] do [0, 1] dla poprawnego wyświetlania
            gen_imgs = (gen_imgs + 1) / 2
            grid = make_grid(gen_imgs, nrow=8)
            saved_images.append((epoch + 1, grid))
        netG.train()

# --------------------
# DIAGNOSTYKA WIZUALNA GAN
# --------------------

# 1. Wykres funkcji strat G i D
plt.figure(figsize=(10, 5))
plt.plot(losses_G, label="Generator Loss")
plt.plot(losses_D, label="Discriminator Loss")
plt.xlabel("Epoka")
plt.ylabel("BCE Loss")
plt.title(f"Historia strat podczas treningu GAN ({DATASET_TYPE})")
plt.legend()
plt.show()

# 2. Wyświetlenie próbek na różnych etapach treningu
fig, axes = plt.subplots(len(saved_images), 1, figsize=(8, 3 * len(saved_images)))
if len(saved_images) == 1:
    axes = [axes]

for ax, (epoch_num, grid) in zip(axes, saved_images):
    ax.imshow(np.transpose(grid.numpy(), (1, 2, 0)), cmap="gray")
    ax.set_title(f"Wygenerowane próbki – Epoka {epoch_num}")
    ax.axis("off")

plt.tight_layout()
plt.show()
