import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


def dct_2d(image):
    M, N = image.shape
    dct_coefficient = np.zeros(image.shape, dtype=np.float32)
    # create index matrix
    x, y = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
    for u in tqdm(range(M)):
        for v in range(N):
            factor_u = np.sqrt(1 / 2) if u == 0 else 1
            factor_v = np.sqrt(1 / 2) if v == 0 else 1

            # caculate cos part
            cos_x = np.cos(((2 * x + 1) * u * np.pi) / (2 * M))
            cos_y = np.cos(((2 * y + 1) * v * np.pi) / (2 * N))

            # sum up and multliply C(u) and C(v)
            dct_coefficient[u, v] = factor_u * factor_v * np.sum(image * cos_x * cos_y)

    dct_coefficient = dct_coefficient * 2 / N
    return dct_coefficient


def idct_2d(coefficient):
    M, N = coefficient.shape
    reconsruted_image = np.zeros(coefficient.shape, dtype=np.float32)
    # create index matrix
    u, v = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
    factor_u = np.where(u == 0, 1 / np.sqrt(2), 1)
    factor_v = np.where(v == 0, 1 / np.sqrt(2), 1)
    for x in tqdm(range(M)):
        for y in range(N):
            cos_u = np.cos(((2 * x + 1) * u * np.pi) / (2 * M))
            cos_v = np.cos(((2 * y + 1) * v * np.pi) / (2 * N))
            reconsruted_image[x][y] = (
                2 / N * np.sum(coefficient * factor_u * factor_v * cos_u * cos_v)
            )
    return reconsruted_image


def dct_1d(signal):
    N = len(signal)
    x = np.array([i for i in range(N)])
    dct_cofficient = np.zeros(N, dtype=np.float32)
    for u in range(N):
        cos_x = np.cos(((2 * x + 1) * u * np.pi) / (2 * N))
        factor = 1 / np.sqrt(2) if u == 0 else 1
        dct_cofficient[u] = factor * np.sum(signal * cos_x)
    dct_cofficient = dct_cofficient * np.sqrt(2 / N)
    return dct_cofficient


image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float32)

# resize to speed up
new_size = (256, 256)
image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

# 2D DCT
print("2D DCT start")
start_time = time.time()
dct_2d = dct_2d(image)
time_2d = time.time() - start_time
print(f"2D DCT end, use:{time_2d:.4f} sec")

# 1D DCT
print("1D DCT start")
start_time = time.time()
dct_1d_row = np.array([dct_1d(row) for row in tqdm(image)])
dct_1d = np.array([dct_1d(col) for col in tqdm(dct_1d_row.T)]).T
time_1d = time.time() - start_time
print(f"1D DCT end, use:{time_1d:.4f} sec")

# reconstruct using 2D IDCT
print("2D IDCT start")
reconstructed_image = idct_2d(dct_2d)
mse = np.mean((image - reconstructed_image) ** 2)
mse = max(mse, 1)
psnr = 10 * np.log10(255**2 / mse)

print("------output------")
print(f"2D DCT use:{time_2d:.4f} sec")
print(f"1D DCT use:{time_1d:.4f} sec")
print(f"PSNR of reconstructed_image = {psnr:.4f}")

plt.imshow(np.log(np.abs(dct_2d) + 1), cmap="gray")
plt.title("DCT Coefficients in Log Domain")
plt.colorbar()
plt.show()
