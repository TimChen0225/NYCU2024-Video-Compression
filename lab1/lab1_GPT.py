import cv2
import numpy as np

# 讀取圖片
image = cv2.imread("lena.png")

# 確認圖片已正確讀取
if image is None:
    print("Error: Unable to read image.")
    exit()

# 提取 R、G、B 通道
B, G, R = cv2.split(image)

# 初始化 Y, U, V, Cb, Cr 通道矩陣
Y = np.zeros_like(R, dtype=float)
U = np.zeros_like(R, dtype=float)
V = np.zeros_like(R, dtype=float)
Cb = np.zeros_like(R, dtype=float)
Cr = np.zeros_like(R, dtype=float)

# 手動計算 YUV 和 YCbCr
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        r = float(R[i, j])
        g = float(G[i, j])
        b = float(B[i, j])

        # RGB -> YUV
        Y[i, j] = 0.299 * r + 0.587 * g + 0.114 * b
        U[i, j] = -0.169 * r - 0.331 * g + 0.5 * b + 128
        V[i, j] = 0.5 * r - 0.419 * g - 0.081 * b + 128

        # RGB -> YCbCr (using BT.601)
        Y[i, j] = 0.299 * r + 0.587 * g + 0.114 * b
        Cb[i, j] = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
        Cr[i, j] = 0.5 * r - 0.418688 * g - 0.081312 * b + 128


# 正規化到 [0, 255] 範圍並轉換為無符號 8 位元格式
def normalize_and_convert(img):
    img = np.clip(img, 0, 255)  # 確保數值在 0 到 255 之間
    return img.astype(np.uint8)


# 將結果正規化並轉換為灰階圖
R = normalize_and_convert(R)
G = normalize_and_convert(G)
B = normalize_and_convert(B)
Y = normalize_and_convert(Y)
U = normalize_and_convert(U)
V = normalize_and_convert(V)
Cb = normalize_and_convert(Cb)
Cr = normalize_and_convert(Cr)

# 儲存 R、G、B、Y、U、V、Cb、Cr 圖片
cv2.imwrite("R_channel.png", R)
cv2.imwrite("G_channel.png", G)
cv2.imwrite("B_channel.png", B)
cv2.imwrite("Y_channel.png", Y)
cv2.imwrite("U_channel.png", U)
cv2.imwrite("V_channel.png", V)
cv2.imwrite("Cb_channel.png", Cb)
cv2.imwrite("Cr_channel.png", Cr)

print("Images saved successfully.")
