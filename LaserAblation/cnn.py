import torch
from torch import nn
class LaserNet3D(nn.Module):
    def __init__(self, R, A, S):
        super().__init__()
        self.R, self.A, self.S = R, A, S

        self.conv = nn.Sequential(
            nn.Conv3d(1, 10, 3, padding=1),
            nn.InstanceNorm3d(10),
            nn.SiLU(),

            nn.Conv3d(10, 20, 3, stride=2, padding=1),
            nn.InstanceNorm3d(20),
            nn.SiLU(),

            nn.Conv3d(20, 40, 3, stride=2, padding=1),
            nn.InstanceNorm3d(40),
            nn.SiLU(),

            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Flatten(),
            nn.Dropout(0.3)
        ).cuda()

        self.fc = nn.Linear(40 * 4 * 4 * 4, 4).cuda()

        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.bias.zero_()

            # Start near middle, end offset slightly in +R direction
            self.fc.bias[:] = torch.tensor([
                -0.5,
                -0.5,
                +0.5,
                +0.5
            ])

    def forward(self, x):
        f = self.conv(x)
        pts = torch.tanh(self.fc(f))

        r0 = (pts[:, 0] + 1) * 0.5 * (self.R-1)
        a0 = (pts[:, 1] + 1) * 0.5 * (self.A-1)

        r1 = (pts[:, 2] + 1) * 0.5 * (self.R-1)
        a1 = (pts[:, 3] + 1) * 0.5 * (self.A-1)

        return r0, a0, r1, a1
