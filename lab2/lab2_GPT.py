import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


# 定義 1D-DCT 函數
def dct_1d(signal):
    N = len(signal)
    dct_coeff = np.zeros(N)
    for k in range(N):
        for n in range(N):
            dct_coeff[k] += signal[n] * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        if k == 0:
            dct_coeff[k] *= np.sqrt(1 / N)
        else:
            dct_coeff[k] *= np.sqrt(2 / N)
    return dct_coeff


# 定義 1D-IDCT 函數
def idct_1d(dct_coeff):
    N = len(dct_coeff)
    signal = np.zeros(N)
    for n in range(N):
        for k in range(N):
            factor = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
            signal[n] += (
                factor * dct_coeff[k] * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
            )
    return signal


# 定義 2D-DCT 函數
def dct_2d(image):
    # 對每一列進行 1D-DCT
    print("正在計算 2D-DCT：第一步 - 對每列進行 1D-DCT")
    step1 = np.array([dct_1d(col) for col in tqdm(image.T)])

    # 對每一行進行 1D-DCT
    print("正在計算 2D-DCT：第二步 - 對每行進行 1D-DCT")
    step2 = np.array([dct_1d(row) for row in tqdm(step1.T)])

    return step2


# 定義 2D-IDCT 函數
def idct_2d(dct_image):
    # 對每一列進行 1D-IDCT
    print("正在計算 2D-IDCT：第一步 - 對每列進行 1D-IDCT")
    step1 = np.array([idct_1d(col) for col in tqdm(dct_image.T)])

    # 對每一行進行 1D-IDCT
    print("正在計算 2D-IDCT：第二步 - 對每行進行 1D-IDCT")
    step2 = np.array([idct_1d(row) for row in tqdm(step1.T)])

    return step2


# 讀取 lena 圖片，並轉換為灰度圖
print("讀取 lena 圖片...")
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float32)

new_size = (128, 128)
image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

# 計算 2D-DCT 並記錄時間
print("開始計算 2D-DCT...")
start_time = time.time()
dct_image = dct_2d(image)
time_2d_dct = time.time() - start_time
print(f"2D-DCT 計算完成，耗時：{time_2d_dct:.4f} 秒")

# 可視化 DCT 係數（對數域）
print("可視化 DCT 係數...")
plt.imshow(np.log(np.abs(dct_image) + 1), cmap="gray")
plt.title("DCT Coefficients in Log Domain")
plt.colorbar()
plt.show()

# 計算 2D-IDCT 並重建圖像
print("開始計算 2D-IDCT 以重建圖像...")
reconstructed_image = idct_2d(dct_image)

# 計算 PSNR
print("計算 PSNR...")
mse = np.mean((image - reconstructed_image) ** 2)
psnr = 10 * np.log10(255**2 / mse)
print(f"PSNR: {psnr:.2f} dB")

# 使用兩次 1D-DCT 的方法計算 DCT 並記錄時間
print("開始計算使用兩次 1D-DCT 的方法...")
start_time = time.time()
dct_image_1d = np.array([dct_1d(row) for row in tqdm(image)])
dct_image_1d = np.array([dct_1d(col) for col in tqdm(dct_image_1d.T)]).T
time_1d_dct = time.time() - start_time
print(f"使用兩次 1D-DCT 計算完成，耗時：{time_1d_dct:.4f} 秒")

# 比較運行時間
print(f"2D-DCT runtime: {time_2d_dct:.4f} seconds")
print(f"Two 1D-DCT runtime: {time_1d_dct:.4f} seconds")

# 結論：
# 可以看到，通過兩次 1D-DCT 計算 2D-DCT 相較於直接進行 2D-DCT 更加快速。
