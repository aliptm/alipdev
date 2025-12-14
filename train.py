# import torch
# from torch.utils.data import DataLoader
# from dataset import VoxelDataset
# from gan import Generator, Discriminator

# EPOCHS = 100
# BATCH_SIZE = 4
# Z_DIM = 128
# LR = 0.0002

# dataset = VoxelDataset("data/objs")
# loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# G = Generator(Z_DIM)
# D = Discriminator()

# opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
# opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

# for epoch in range(EPOCHS):
#     for real in loader:
#         batch = real.size(0)
#         real = real * 2 - 1  # [-1, 1]

#         # ---- Train D ----
#         z = torch.randn(batch, Z_DIM)
#         fake = G(z)

#         loss_D = (
#             torch.mean((D(real) - 1) ** 2) +
#             torch.mean(D(fake.detach()) ** 2)
#         )

#         opt_D.zero_grad()
#         loss_D.backward()
#         opt_D.step()

#         # ---- Train G ----
#         loss_G = torch.mean((D(fake) - 1) ** 2)

#         opt_G.zero_grad()
#         loss_G.backward()
#         opt_G.step()

#     print(f"Epoch {epoch+1}/{EPOCHS}  D:{loss_D.item():.3f}  G:{loss_G.item():.3f}")

# torch.save(G.state_dict(), "generator.pth")
import torch
from torch.utils.data import DataLoader
from dataset import VoxelDataset
from gan import Generator, Discriminator

EPOCHS = 30
BATCH_SIZE = 2
Z_DIM = 64
LR = 0.0003

dataset = VoxelDataset("data/objs")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

G = Generator(Z_DIM)
D = Discriminator()

opt_G = torch.optim.Adam(G.parameters(), lr=LR)
opt_D = torch.optim.Adam(D.parameters(), lr=LR)

for epoch in range(EPOCHS):
    for real in loader:
        batch = real.size(0)
        real = real * 2 - 1

        z = torch.randn(batch, Z_DIM)
        fake = G(z)

        loss_D = ((D(real) - 1) ** 2).mean() + (D(fake.detach()) ** 2).mean()
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        loss_G = ((D(fake) - 1) ** 2).mean()
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch {epoch+1}/{EPOCHS}  D:{loss_D.item():.3f}  G:{loss_G.item():.3f}")

torch.save(G.state_dict(), "generator.pth")
