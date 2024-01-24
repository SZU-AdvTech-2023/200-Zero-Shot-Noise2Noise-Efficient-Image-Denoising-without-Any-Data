import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义图像转换操作，将图像转换为 PyTorch 张量
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为 PyTorch 张量，包含透明通道
])


class SMU(nn.Module):
    def __init__(self, alpha=0.25):
        super(SMU, self).__init__()
        self.alpha = alpha
        # initialize mu
        self.mu = torch.nn.Parameter(torch.tensor(1000000.0))

    def forward(self, x):
        return ((1 + self.alpha) * x + (1 - self.alpha) * x * torch.erf(self.mu * (1 - self.alpha) * x)) / 2

class network(nn.Module):
    def __init__(self, n_chan, chan_embed=48):
        super(network, self).__init__()
        self.act=SMU()
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)

        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


def pair_downsampler(img):
    # img has shape B C H W
    c = img.shape[1]

    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device) #(1,1,2,2)
    filter1 = filter1.repeat(c, 1, 1, 1) #(c,1,2,2)

    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2  #(1,3,122,122)


def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt, pred)


def loss_func(noisy_img,model):
    noisy1, noisy2 = pair_downsampler(noisy_img)

    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)

    loss_res = 1 / 2 * (mse(noisy1, pred2) + mse(noisy2, pred1))

    noisy_denoised = noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)

    loss_cons = 1 / 2 * (mse(pred1, denoised1) + mse(pred2, denoised2))

    loss = loss_res + loss_cons

    return loss


def train(model, optimizer, noisy_img):
    loss = loss_func(noisy_img,model)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, noisy_img, clean_img):
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)
        MSE = mse(clean_img, pred).item()
        PSNR = 10 * np.log10(1 / MSE)

    return PSNR


def denoise(model, noisy_img):
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)

    return pred

max_epoch = 2000   # training epochsS
lr = 0.001           # learning rate
step_size = 1500     # number of epochs at which learning rate decays
gamma = 0.5          # factor by which learning rate decays


def Image_denoise(noisy_image_path):

    image = Image.open(noisy_image_path)

    image = image.convert("RGB")

    noisy_img = transform(image)
    noisy_img = torch.unsqueeze(noisy_img, 0)  # 在第一维上添加一个维度
    device = 'cuda'
    noisy_img = noisy_img.to(device)

    n_chan = noisy_img.shape[1]
    model = network(n_chan)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    for epoch in tqdm(range(max_epoch)):
        optimizer.zero_grad()
        loss_item = train(model, optimizer, noisy_img)
        # 使用学习率调度器优化学习率
        scheduler.step()
    denoised_img = denoise(model, noisy_img)

    # 将输出的张量转换为NumPy数组，并从GPU移到CPU
    denoised_np = denoised_img.cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
    denoised_np = (denoised_np - denoised_np.min()) / (denoised_np.max() - denoised_np.min())
    output_folder = "output"  # 替换为你想要保存的文件夹路径
    if not os.path.exists(output_folder) :
        os.makedirs(output_folder)
    output_filename = "denoise_img.png"
    output_path = os.path.join(output_folder, output_filename)

    # 保存图像为文件
    plt.imsave(output_path, denoised_np)

