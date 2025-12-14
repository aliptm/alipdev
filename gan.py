# import torch.nn as nn

# # ---------------- Generator ----------------
# class Generator(nn.Module):
#     def __init__(self, z_dim=128):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.ConvTranspose3d(z_dim, 128, 4, 1, 0),
#             nn.BatchNorm3d(128),
#             nn.ReLU(),

#             nn.ConvTranspose3d(128, 64, 4, 2, 1),
#             nn.BatchNorm3d(64),
#             nn.ReLU(),

#             nn.ConvTranspose3d(64, 32, 4, 2, 1),
#             nn.BatchNorm3d(32),
#             nn.ReLU(),

#             nn.ConvTranspose3d(32, 1, 4, 2, 1),
#             nn.Tanh()
#         )

#     def forward(self, z):
#         return self.net(z.view(z.size(0), z.size(1), 1, 1, 1))


# # ---------------- Discriminator ----------------
# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Conv3d(1, 32, 4, 2, 1),
#             nn.LeakyReLU(0.2),

#             nn.Conv3d(32, 64, 4, 2, 1),
#             nn.BatchNorm3d(64),
#             nn.LeakyReLU(0.2),

#             nn.Conv3d(64, 128, 4, 2, 1),
#             nn.BatchNorm3d(128),
#             nn.LeakyReLU(0.2),

#             nn.Conv3d(128, 1, 4, 1, 0)
#         )

#     def forward(self, x):
#         return self.net(x).view(-1, 1)
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(z_dim, 64, 4, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 1, 4, 1, 0)
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)
