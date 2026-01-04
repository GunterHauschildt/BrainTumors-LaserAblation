import torch
import torch.nn as nn
import torch.optim as optim
import cv2 as cv2
import numpy as np

H, W = 256, 256


class LineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.InstanceNorm2d(16),
            nn.SiLU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.SiLU(),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        ).cuda()
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 64),
            nn.Linear(64, 2)
        ).cuda()

    def forward(self, x):
        x = self.conv(x)
        out = self.fc(x)
        pts = torch.tanh(out)  # now in [-1,1]
        x0 = (pts[:, 0] + 1) * 0.5 * (W - 1)
        x1 = (pts[:, 1] + 1) * 0.5 * (W - 1)
        return x0, x1


def random_blob():

    BLOB_REWARD = 4.0
    BACK_REWARD = -4.0

    blob = np.ones((H, W), dtype=np.float32) * BACK_REWARD

    # sample axes first
    ax = np.random.randint(W // 12, W // 8)
    ay = np.random.randint(H // 20, H // 16)

    margin_x = W // 4
    margin_y = H // 4

    cx = np.random.randint(margin_x, W - margin_x)
    cy = np.random.randint(margin_y, H - margin_y)

    center = (cx, cy)
    axes = (ax, ay)
    angle = np.random.rand() * 360

    cv2.ellipse(blob, center, axes, angle, 0.0, 360.0, BLOB_REWARD, -1)

    blob_tensor = torch.tensor(blob, dtype=torch.float32, device='cuda')[None, None, :, :]
    return (blob * 255).astype(np.uint8), blob_tensor


def line_tensor(x0, y0, x1, y1, sigma=0.66):
    segment_smooth = 0.25
    yv, xv = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    xv = xv.float().cuda()
    yv = yv.float().cuda()

    # line direction
    dx = x1 - x0
    dy = y1 - y0
    norm = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
    nx = dy / norm
    ny = -dx / norm

    # distance to infinite line
    dist = nx * (xv - x0) + ny * (yv - y0)
    mask_line = torch.exp(-(dist ** 2) / (2 * sigma ** 2))

    # parametric t along line
    t = ((xv - x0) * dx + (yv - y0) * dy) / (dx ** 2 + dy ** 2 + 1e-8)

    # soft segment mask using smooth clipping
    mask_segment = torch.sigmoid((t) / segment_smooth) * torch.sigmoid((1 - t) / segment_smooth)

    mask = mask_line * mask_segment
    return mask

# Initialize
net = LineNet()
net.train()
opt = optim.Adam(net.parameters(), lr=0.0001)

# Training loop
for step in range(200000):
    blob, blob_t = random_blob()

    x0, x1 = net(blob_t)
    y0 = torch.tensor(0, dtype=torch.float32, device='cuda')
    y1 = torch.tensor(H - 1, dtype=torch.float32, device='cuda')

    line_t = line_tensor(x0, y0, x1, y1, 10.0)
    loss = -(line_t * blob_t).sum()

    line_np = line_t.detach().cpu().numpy()

    opt.zero_grad()
    loss.backward()
    opt.step()

    # visualization every 10 steps
    if step % 100 == 0:
        line = (line_t.detach().cpu().numpy() * 255).astype(np.uint8)
        contours, _ = cv2.findContours(line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        blob_draw = cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR)
        blob_draw = cv2.drawContours(blob_draw, contours, -1, (255, 0, 255))
        print(f"Step {step}, Loss {loss.item():.3f}")  #, {int(x0)}, {int(y0)}, {int(x1)}, {int(y1)}")
        cv2.imshow("Blob", blob_draw)
        cv2.moveWindow("Blob", 200, 200)
        cv2.resizeWindow("Blob", 512, 512)
        cv2.waitKey(1)

pass
cv2.destroyAllWindows()