import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset

S = 1e-5

# ===============================
# Learnable JPEG Compression Model
# ===============================


class LearnableJPEG(nn.Module):
    def __init__(self, image_shape):
        super(LearnableJPEG, self).__init__()
        self.image_shape = image_shape

        # Learnable quantization tables for luminance (Y) and chrominance (C)
        # Initial luminance quantization table
        self.q_luminance = nn.Parameter(torch.empty(8, 8))
        torch.nn.init.uniform_(self.q_luminance, a=1 * S, b=2 * S)
        # Initial chrominance quantization table
        self.q_chrominance = nn.Parameter(torch.empty(8, 8))
        torch.nn.init.uniform_(self.q_chrominance, a=1 * S, b=2 * S)

        # Constants for DCT calculation
        self.dct_basis = self.create_dct_basis(8)
        self.dct_basis.requires_grad = False

    def forward(self, x):
        # Shape check before processing
        if x.ndim != 4 or x.size(1) != 3:
            raise ValueError(
                f"Input tensor shape must be [B, 3, H, W], but got {x.shape}"
            )

        # Convert to YCbCr
        y, cb, cr = self.rgb_to_ycbcr(x)
        # save_image(y, "test_y.jpg")
        # save_image(cb, "test_cb.jpg")
        # save_image(cr, "test_cr.jpg")

        # Blockwise DCT
        y_dct = self.blockwise_dct(y)
        cb_dct = self.blockwise_dct(cb)
        cr_dct = self.blockwise_dct(cr)

        # Quantization
        y_quantized = torch.round(y_dct / (self.q_luminance / S))
        cb_quantized = torch.round(cb_dct / (self.q_chrominance / S))
        cr_quantized = torch.round(cr_dct / (self.q_chrominance / S))

        # Dequantization
        y_recon = y_quantized * (self.q_luminance / S)
        cb_recon = cb_quantized * (self.q_chrominance / S)
        cr_recon = cr_quantized * (self.q_chrominance / S)

        # Blockwise IDCT
        y_rec = self.blockwise_idct(y_recon)
        cb_rec = self.blockwise_idct(cb_recon)
        cr_rec = self.blockwise_idct(cr_recon)

        # Convert back to RGB
        x_rec = self.ycbcr_to_rgb(y_rec, cb_rec, cr_rec)

        return x_rec

    def create_dct_basis(self, block_size):
        """Generate DCT basis functions for a block size."""
        basis = torch.zeros((block_size, block_size, block_size, block_size))
        for u in range(block_size):
            for v in range(block_size):
                for x in range(block_size):
                    for y in range(block_size):
                        basis[u, v, x, y] = self.dct_coeff(
                            u, x, block_size
                        ) * self.dct_coeff(v, y, block_size)
        return basis

    def dct_coeff(self, k, n, N):
        """Calculate DCT coefficient."""
        if k == 0:
            return np.sqrt(1 / N)
        else:
            return np.sqrt(2 / N) * np.cos(np.pi * k * (2 * n + 1) / (2 * N))

    def blockwise_dct(self, x):
        """Perform blockwise DCT on an image."""
        block_size = 8
        x_unf = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        x_unf = x_unf.contiguous().view(
            x.size(0), x.size(1), -1, block_size, block_size
        )
        dct_blocks = torch.einsum("bijmn,uvmn->bijuv", x_unf, self.dct_basis)
        return dct_blocks

    def blockwise_idct(self, x):
        """Perform blockwise IDCT on an image."""
        block_size = 8
        idct_blocks = torch.einsum("bijuv,uvmn->bijmn", x, self.dct_basis)
        idct_blocks = idct_blocks.view(
            x.size(0), x.size(1), 32, 32, block_size, block_size
        )
        idct_blocks = idct_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        idct_blocks = idct_blocks.view(x.size(0), x.size(1), 256, 256)
        return idct_blocks

    def rgb_to_ycbcr(self, x):
        """Convert RGB to YCbCr color space."""
        ycbcr = torch.empty_like(x)
        ycbcr[:, 0, :, :] = (
            x[:, 0, :, :] * 0.299 + x[:, 1, :, :] * 0.587 + x[:, 2, :, :] * 0.114
        )
        ycbcr[:, 1, :, :] = (
            x[:, 0, :, :] * (-0.168736)
            + x[:, 1, :, :] * (-0.331264)
            + x[:, 2, :, :] * 0.5
            + 128
        )
        ycbcr[:, 2, :, :] = (
            x[:, 0, :, :] * 0.5
            + x[:, 1, :, :] * (-0.418688)
            + x[:, 2, :, :] * (-0.081312)
            + 128
        )

        # === test ===
        # save_image(ycbcr[:, 0:1, :, :] / 255.0, "before_y.jpg")
        # save_image((ycbcr[:, 1:2, :, :] - 16) / 224.0, "before_cb.jpg")
        # save_image((ycbcr[:, 2:3, :, :] - 16) / 224.0, "before_cr.jpg")
        # print("x shape:", x.shape)
        # print("ycbcr shape", ycbcr.shape)
        # print("x range", x.min().item(), x.max().item())
        # print("Y range:", ycbcr[:, 0:1].min().item(), ycbcr[:, 0:1].max().item())
        # print("Cb range:", ycbcr[:, 1:2].min().item(), ycbcr[:, 1:2].max().item())
        # print("Cr range:", ycbcr[:, 2:3].min().item(), ycbcr[:, 2:3].max().item())

        return ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]

    def ycbcr_to_rgb(self, y, cb, cr):
        """Convert YCbCr back to RGB color space."""
        matrix = torch.tensor(
            [[1.0, 0.0, 1.402], [1.0, -0.344136, -0.714136], [1.0, 1.772, 0.0]],
            device=y.device,
        )
        # shift = torch.tensor([0, 128, 128], device=y.device).view(1, 3, 1, 1)
        ycbcr = torch.cat([y, cb, cr], dim=1)
        ycbcr[:, 1:, :, :] -= 128

        rgb = torch.empty_like(ycbcr)
        rgb[:, 0, :, :] = ycbcr[:, 0, :, :] * 1 + ycbcr[:, 2, :, :] * 1.402
        rgb[:, 1, :, :] = (
            ycbcr[:, 0, :, :] * 1
            + ycbcr[:, 1, :, :] * (-0.344136)
            + ycbcr[:, 2, :, :] * (-0.714136)
        )
        rgb[:, 2, :, :] = ycbcr[:, 0, :, :] * 1 + ycbcr[:, 1, :, :] * 1.722

        # === test ===
        # save_image(ycbcr[:, 0, :, :] / 255.0, "after_y.jpg")
        # save_image((ycbcr[:, 1, :, :] - 16) / 224.0, "after_cb.jpg")
        # save_image((ycbcr[:, 2, :, :] - 16) / 224.0, "after_cr.jpg")
        # save_image(rgb[:, 0, :, :] / 255.0, "r.jpg")
        # save_image(rgb[:, 1, :, :] / 255.0, "g.jpg")
        # save_image(rgb[:, 2, :, :] / 255.0, "b.jpg")
        # print("Y range:", ycbcr[:, 0:1].min().item(), ycbcr[:, 0:1].max().item())
        # print("Cb range:", ycbcr[:, 1:2].min().item(), ycbcr[:, 1:2].max().item())
        # print("Cr range:", ycbcr[:, 2:3].min().item(), ycbcr[:, 2:3].max().item())
        return torch.clamp(rgb, 0, 255)


# ===============================
# Loss Functions
# ===============================


def distortion_loss(x, x_rec):
    mse = torch.mean((x - x_rec) ** 2)
    lpips = 0  # Placeholder for LPIPS loss (can integrate perceptual loss libraries)
    return mse + 0.01 * lpips


def rate_loss(q_luminance, q_chrominance):
    return torch.sum(torch.abs(q_luminance)) + torch.sum(torch.abs(q_chrominance))


def total_loss(x, x_rec, q_luminance, q_chrominance, lambda_rate):
    d_loss = distortion_loss(x, x_rec)
    r_loss = rate_loss(q_luminance, q_chrominance)
    return d_loss + lambda_rate * r_loss


# ===============================
# Load Image and Transform
# ===============================


def load_image(image_path, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255),  # Scale to [0, 255]
        ]
    )
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


class ImageDataset(Dataset):
    def __init__(self, image_path, image_size):
        self.image = load_image(image_path, image_size)

    def __len__(self):
        return 1000  # 重複 1000 次

    def __getitem__(self, idx):
        return self.image.squeeze(0)  # 移除 batch 維度


# ===============================
# Train
# ===============================
def train(model, dataloader, optimizer, epochs, lambda_rate, device):
    model.train()
    for epoch in range(epochs):
        total_epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)  # 將 batch 移到 GPU/CPU

            # 前向傳播
            output_image = model(batch)

            # 計算損失
            loss = total_loss(
                batch, output_image, model.q_luminance, model.q_chrominance, lambda_rate
            )

            # 反向傳播與優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {total_epoch_loss / len(dataloader):.4f}"
        )


# ===============================
# Main
# ===============================

if __name__ == "__main__":
    # 設定設備與模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LearnableJPEG(image_shape=(1, 3, 256, 256)).to(device)

    # 訓練參數
    epochs = 10
    lambda_rate = 1e-4
    learning_rate = 1e-6

    # 優化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 建立 DataLoader
    dataset = ImageDataset(image_path="lena.png", image_size=256)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 開始訓練
    train(model, dataloader, optimizer, epochs, lambda_rate, device)

    # 儲存模型與量化表
    print("Training complete.")
    print("Learned Luminance Quantization Table:")
    print(model.q_luminance.detach().cpu().numpy())
    print("Learned Chrominance Quantization Table:")
    print(model.q_chrominance.detach().cpu().numpy())
    torch.save(model.state_dict(), "learnable_jpeg_model.pth")
