import numpy as np
import torch
import torch.nn as nn
import cv2

# Minimal SRCNN architecture (PyTorch version)
class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Load pre-trained weights from .pth file
def load_pretrained_srcnn(weights_path="srcnn_x4.pth", device=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(weights_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    return model, device

# Enhance a BGR image using SRCNN (PyTorch)
def enhance_with_srcnn(img_bgr: np.ndarray, model=None, device=None) -> np.ndarray:
    # Convert BGR to YCrCb
    img_y_cr_cb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    y = y.astype(np.float32) / 255.0
    y_tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_tensor = y_tensor.to(device)
    if model is None:
        model, device = load_pretrained_srcnn(device=device)
    with torch.no_grad():
        y_enhanced = model(y_tensor)
    y_enhanced = y_enhanced.squeeze().cpu().numpy()
    y_enhanced = np.clip(y_enhanced, 0, 1) * 255.0
    y_enhanced = y_enhanced.astype(np.uint8)
    img_y_cr_cb = cv2.merge([y_enhanced, cr, cb])
    img_bgr_enhanced = cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCrCb2BGR)
    return img_bgr_enhanced 