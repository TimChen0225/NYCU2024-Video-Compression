import numpy as np
import cv2
import time
from skimage.metrics import peak_signal_noise_ratio as psnr

block_size = 8


def full_search(reference_frame, current_frame, search_range):
    h, w = reference_frame.shape
    block_size = 8
    motion_vectors = np.zeros((h // block_size, w // block_size, 2), dtype=np.int32)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            best_match = None
            min_error = float("inf")
            current_block = current_frame[i : i + block_size, j : j + block_size]

            for di in range(-search_range, search_range + 1):
                for dj in range(-search_range, search_range + 1):
                    ref_i = i + di
                    ref_j = j + dj

                    if 0 <= ref_i <= h - block_size and 0 <= ref_j <= w - block_size:
                        ref_block = reference_frame[
                            ref_i : ref_i + block_size, ref_j : ref_j + block_size
                        ]
                        error = np.sum(np.abs(current_block - ref_block))

                        if error < min_error:
                            min_error = error
                            best_match = (di, dj)

            motion_vectors[i // block_size, j // block_size] = best_match

    return motion_vectors


def motion_compensation(reference_frame, motion_vectors):
    h, w = reference_frame.shape
    block_size = 8
    compensated_frame = np.zeros_like(reference_frame)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            di, dj = motion_vectors[i // block_size, j // block_size]
            ref_i = i + di
            ref_j = j + dj
            compensated_frame[i : i + block_size, j : j + block_size] = reference_frame[
                ref_i : ref_i + block_size, ref_j : ref_j + block_size
            ]

    return compensated_frame


def normalize_residual(residual):
    # 將 residual 映射到 0-255 之間
    min_val = np.min(residual)
    max_val = np.max(residual)
    print(f"Residual min: {min_val}, max: {max_val}")

    normalized_residual = (residual - min_val) / (max_val - min_val)
    normalized_residual = 255 * normalized_residual
    normalized_residual = normalized_residual.astype(np.uint8)

    return normalized_residual


def three_step_search(current_block, reference_frame, current_block_pos, search_range):
    # 初始化搜索範圍
    step_size = search_range // 2
    best_mv = (0, 0)
    min_cost = float("inf")

    while step_size > 0:
        for y in range(-step_size, step_size + 1, step_size):
            for x in range(-step_size, step_size + 1, step_size):
                ref_y = current_block_pos[0] + best_mv[0] + y
                ref_x = current_block_pos[1] + best_mv[1] + x

                # 確保索引在範圍內
                if (
                    0 <= ref_y < reference_frame.shape[0] - block_size
                    and 0 <= ref_x < reference_frame.shape[1] - block_size
                ):
                    reference_block = reference_frame[
                        ref_y : ref_y + block_size, ref_x : ref_x + block_size
                    ]
                    # 使用 SAD 作為匹配誤差
                    cost = np.sum(np.abs(current_block - reference_block))

                    if cost < min_cost:
                        min_cost = cost
                        best_mv = (best_mv[0] + y, best_mv[1] + x)

        step_size //= 2

    return best_mv


def run_me_mc(reference_frame, current_frame, search_range, algorithm="full"):
    if algorithm == "full":
        start_time = time.time()
        motion_vectors = full_search(reference_frame, current_frame, search_range)
        end_time = time.time()
    else:
        # 插入其他搜索算法（如三步搜索）的邏輯
        start_time = time.time()
        motion_vectors = np.zeros(
            (
                reference_frame.shape[0] // block_size,
                reference_frame.shape[1] // block_size,
                2,
            ),
            dtype=np.int32,
        )
        for i in range(0, current_frame.shape[0], block_size):
            for j in range(0, current_frame.shape[1], block_size):
                current_block = current_frame[i : i + block_size, j : j + block_size]
                mv = three_step_search(
                    current_block, reference_frame, (i, j), search_range
                )
                motion_vectors[i // block_size, j // block_size] = mv
        end_time = time.time()

    # 進行運動補償，生成重建幀
    reconstructed_frame = motion_compensation(reference_frame, motion_vectors)

    # 計算殘差
    residual = current_frame - reconstructed_frame

    # 計算 PSNR 值
    psnr_value = psnr(current_frame, reconstructed_frame)

    # 計算執行時間
    runtime = end_time - start_time

    # 保存重建幀和原始殘差
    cv2.imwrite(
        f"reconstructed_frame_{algorithm}_sr{search_range}.png", reconstructed_frame
    )

    # 將殘差轉換為可視化的灰階圖片
    normalized_residual = normalize_residual(residual)
    cv2.imwrite(f"residual_{algorithm}_sr{search_range}.png", residual)
    cv2.imwrite(
        f"normalized_residual_{algorithm}_sr{search_range}.png", normalized_residual
    )

    return psnr_value, runtime, reconstructed_frame, residual


# 測試範例
if __name__ == "__main__":
    # 讀取兩個相鄰幀
    reference_frame = cv2.imread("one_gray.png", cv2.IMREAD_GRAYSCALE)
    current_frame = cv2.imread("two_gray.png", cv2.IMREAD_GRAYSCALE)

    # 設定不同的搜索範圍
    search_ranges = [8, 16, 32]

    # 進行全域搜索與三步搜索的比較
    for search_range in search_ranges:
        psnr_full, runtime_full, _, _ = run_me_mc(
            reference_frame, current_frame, search_range, algorithm="full"
        )
        psnr_tss, runtime_tss, _, _ = run_me_mc(
            reference_frame, current_frame, search_range, algorithm="three_step"
        )

        print(f"Search Range: ±{search_range}")
        print(
            f"Full Search: PSNR = {psnr_full:.2f}, Runtime = {runtime_full:.4f} seconds"
        )
        print(
            f"Three-Step Search: PSNR = {psnr_tss:.2f}, Runtime = {runtime_tss:.4f} seconds"
        )
        print("---------------------------------------------------")
