import cv2
import os
import torch
import numpy as np
from PIL import Image

from torchvision import transforms


def fft(x, rate):
    # the smaller rate, the smoother; the larger rate, the darker
    # rate = 4, 8, 16, 32
    mask = torch.zeros(x.shape).to(x.device)
    w, h = x.shape[-2:]
    line = int((w * h * rate) ** .5 // 2)
    mask[:, :, w // 2 - line:w // 2 + line, h // 2 - line:h // 2 + line] = 1

    fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
    # mask[fft.float() > self.freq_nums] = 1
    # high pass: 1-mask, low pass: mask
    fft = fft * (1 - mask)
    # fft = fft * mask
    fr = fft.real
    fi = fft.imag

    fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
    inv = torch.fft.ifft2(fft_hires, norm="forward").real
    inv = torch.abs(inv)

    return inv


file_dir = './test'
file_list = os.listdir(file_dir)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.],
                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                         std=[1, 1, 1]),
    transforms.ToPILImage()
])

for f in file_list:
    if f.endswith('.jpg'):
        file = os.path.join(file_dir, f)
        img = Image.open(file).convert('RGB')
        ts = transform(img)
        ts=fft(ts.unsqueeze(0),0.25)
        r_img = inverse_transform(ts[0])
        tmp = np.asarray(r_img)
        print(np.unique(tmp,return_counts=True))
        r_img.save(file.replace('.jpg','_0.png'))