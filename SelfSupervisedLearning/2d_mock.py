import os.path

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import cv2 as cv2
import numpy as np

H, W = 256, 256
SIGMA_MIN = 2
SIGMA_MAX = 21

class LineNet(nn.Module):

    MODEL_PATH = "SelfSupervisedLearning/2D_mock.pth"

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
            nn.Linear(64, 3)
        ).cuda()

    def forward(self, x):
        x = self.conv(x)
        out = self.fc(x)
        pts_w_sigma = torch.tanh(out)  # now in [-1,1]
        x0 = (pts_w_sigma[:, 0] + 1) * 0.5 * (W - 1)
        x1 = (pts_w_sigma[:, 1] + 1) * 0.5 * (W - 1)
        sigma = SIGMA_MIN + (pts_w_sigma[:, 2] + 1.0) * (SIGMA_MAX - SIGMA_MIN) / 2.0
        return x0, x1, sigma

    def save(self):
        torch.save(self.state_dict(), LineNet.MODEL_PATH)

    def set_weights_if(self):
        if os.path.isfile(LineNet.MODEL_PATH):
            self.load_state_dict(torch.load(LineNet.MODEL_PATH))


def random_blob():

    REWARD = 10.0, (0, 0, 255)
    PENALITY = -10.0, (0, 255, 0)

    blob = np.zeros((H, W), dtype=np.float32)
    blob_draw = np.zeros((H, W, 3), dtype=np.uint8)

    ax = np.random.randint(W // 12, W // 8)
    ay = np.random.randint(H // 20, H // 16)

    margin_x = W // 4
    margin_y = H // 4

    cx = np.random.randint(margin_x, W - margin_x)
    cy = np.random.randint(margin_y, H - margin_y)

    # draw the ellipse (the tumor to be rewarded if we hit it)
    angle = np.random.rand() * 360
    cv2.ellipse(blob, (cx, cy), (ax, ay), angle, 0.0, 360.0, REWARD[0], -1)
    cv2.ellipse(blob_draw, (cx, cy), (ax, ay), angle, 0.0, 360.0, REWARD[1], -1)

    # now draw a 2nd ellipse near the first, offset on only the x-axis
    # (the healthy tissue that will get a penality if we hit it)

    delta_x = np.random.randint(ax * 1, ax * 2)
    delta_y = np.random.randint(-ay * 2, ay * 2)
    angle = np.random.rand() * 360
    ax = np.random.randint(W // 12, W // 8)
    ay = np.random.randint(H // 20, H // 16)
    cv2.ellipse(blob, (cx + delta_x, cy + delta_y), (ax, ay), angle, 0.0, 360.0, PENALITY[0], -1)
    cv2.ellipse(blob_draw, (cx + delta_x, cy + delta_y), (ax, ay), angle, 0.0, 360.0, PENALITY[1], -1)

    delta_x = np.random.randint(ax * 1, ax * 2)
    delta_y = np.random.randint(-ay * 2, ay * 2)
    ax = np.random.randint(W // 12, W // 8)
    ay = np.random.randint(H // 20, H // 16)
    angle = np.random.rand() * 360
    cv2.ellipse(blob, (cx - delta_x, cy + delta_y), (ax, ay), angle, 0.0, 360.0, PENALITY[0], -1)
    cv2.ellipse(blob_draw, (cx - delta_x, cy + delta_y), (ax, ay), angle, 0.0, 360.0, PENALITY[1], -1)

    blob_tensor = torch.tensor(blob, dtype=torch.float32, device='cuda')[None, None, :, :]

    return blob_draw, blob_tensor


def line_tensor(x0, y0, x1, y1, sigma):
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
    mask_segment = torch.sigmoid(t / segment_smooth) * torch.sigmoid((1 - t) / segment_smooth)

    mask = mask_line * mask_segment
    return mask

# Initialize
net = LineNet()
net.train()
net.set_weights_if()

opt = optim.Adam(net.parameters(), lr=0.0001)

# Load up a background. This is simply to make the video look better. That's it.
# (this whole project is about trying to be impressive looking)
background = cv2.imread("SelfSupervisedLearning/background.jpg")

# A video_writer.
video_writer = cv2.VideoWriter(
        "SelfSupervisedLearning/2D_mock.mp4",
        cv2.VideoWriter.fourcc(*'mp4v'),
        1.5,
        (background.shape[1], background.shape[0])
    )
NUM_VIDEO_FRAMES = 100

# Training loop
NUM_TRAINING_FRAMES = 1000
for step in range(NUM_TRAINING_FRAMES + NUM_VIDEO_FRAMES):
    blob_draw, blob_t = random_blob()

    x0, x1, sigma = net(blob_t)
    y0 = torch.tensor(0, dtype=torch.float32, device='cuda')
    y1 = torch.tensor(H - 1, dtype=torch.float32, device='cuda')

    line_t = line_tensor(x0, y0, x1, y1, sigma)
    loss = -(line_t * blob_t).sum()

    line_np = line_t.detach().cpu().numpy()

    opt.zero_grad()
    loss.backward()
    opt.step()

    # visualization every N steps (or when we're writing)
    draw = None
    if step % 100 == 0 or step > NUM_TRAINING_FRAMES:
        line = line_t.detach().cpu().numpy()
        line = np.where(line > .33, 255, 0).astype(np.uint8)
        line = cv2.resize(line, background.shape[:2][::-1])
        line, _ = cv2.findContours(line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        blob_draw = cv2.resize(blob_draw, background.shape[:2][::-1])
        draw = background.copy()
        for ch in [0, 1, 2]:
            draw[:, :, ch] = np.where(blob_draw[:, :, ch] != 0, blob_draw[:, :, ch], draw[:, :, ch])
        draw = cv2.drawContours(draw, line, -1, (255, 100, 100), 3)
        print(f"Step {step}, Loss {loss.item():.3f}")  #, {int(x0)}, {int(y0)}, {int(x1)}, {int(y1)}")
        cv2.imshow("2D Simulation", draw)
        cv2.moveWindow("2D Simulation", 200, 200)
        if step <= NUM_TRAINING_FRAMES:
            cv2.waitKey(1)

    # Save every N
    if step % 1000 == 0:
        net.save()

    # write after we've trained
    if step > NUM_TRAINING_FRAMES and draw is not None:
        if (cv2.waitKey() == ord('g')):
            video_writer.write(draw)

cv2.destroyAllWindows()