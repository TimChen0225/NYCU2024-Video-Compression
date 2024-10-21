import numpy as np
import cv2
import time
import os
from skimage.metrics import peak_signal_noise_ratio as psnr

block_size = 8

path = ".\\output"
if not os.path.isdir(path):
    os.mkdir(path)


def full_search(reference_frame, current_frame, search_range):
    h, w = reference_frame.shape
    motion_vectors = np.zeros((h // block_size, w // block_size, 2), dtype=np.int32)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # print(f"full search:{i},{j}")
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
                        error = np.sum((current_block - ref_block) ** 2)

                        if error < min_error:
                            min_error = error
                            best_match = (di, dj)

            motion_vectors[i // block_size, j // block_size] = best_match

    return motion_vectors


def three_step_search(reference_frame, current_frame, search_range):
    h, w = reference_frame.shape
    motion_vectors = np.zeros((h // block_size, w // block_size, 2), dtype=np.int32)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # print(f"three step:{i},{j}")
            best_move = (0, 0)
            min_error = float("inf")
            search_step = search_range // 2
            current_block = current_frame[i : i + block_size, j : j + block_size]

            while search_step > 0:
                new_move = (0, 0)
                for y in range(-search_step, search_step + 1, search_step):
                    for x in range(-search_step, search_step + 1, search_step):
                        # print(f"three step step:{i},{j} -> {x},{y}")
                        ref_y = i + (best_move[0] + y) * block_size
                        ref_x = j + (best_move[1] + x) * block_size

                        if (
                            0 <= ref_y <= h - block_size
                            and 0 <= ref_x <= w - block_size
                        ):
                            ref_block = reference_frame[
                                ref_y : ref_y + block_size, ref_x : ref_x + block_size
                            ]
                            error = np.sum((current_block - ref_block) ** 2)

                            if error < min_error:
                                min_error = error
                                new_move = (best_move[0] + y, best_move[1] + x)

                best_move = new_move
                search_step = search_step // 2

            best_move_pixel = (best_move[0] * block_size, best_move[1] * block_size)
            motion_vectors[i // block_size, j // block_size] = best_move_pixel

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


def me_mc(reference_frame, current_frame, search_range, algorithm):
    if algorithm == "full":
        start_time = time.time()
        motion_vectors = full_search(reference_frame, current_frame, search_range)
        end_time = time.time()
    else:
        start_time = time.time()
        motion_vectors = three_step_search(reference_frame, current_frame, search_range)
        end_time = time.time()

    reconstructed_frame = motion_compensation(reference_frame, motion_vectors)

    residual = np.clip(
        current_frame.astype(np.int16) - reconstructed_frame.astype(np.int16), 0, 255
    ).astype(np.uint8)

    psnr_value = psnr(current_frame, reconstructed_frame)

    runtime = end_time - start_time

    cv2.imwrite(
        f".\\output\\reconstructed_frame_{algorithm}_sr{search_range}.png",
        reconstructed_frame,
    )

    cv2.imwrite(f".\\output\\residual_{algorithm}_sr{search_range}.png", residual)

    return psnr_value, runtime


if __name__ == "__main__":
    reference_frame = cv2.imread("one_gray.png", cv2.IMREAD_GRAYSCALE)
    current_frame = cv2.imread("two_gray.png", cv2.IMREAD_GRAYSCALE)
    search_ranges = [8, 16, 32]

    for search_range in search_ranges:
        psnr_full, time_full = me_mc(
            reference_frame, current_frame, search_range, algorithm="full"
        )
        psnr_three_step, time_three_step = me_mc(
            reference_frame, current_frame, search_range, algorithm="three_step"
        )

        print(f"Search Range: ±{search_range}")
        print(f"Full Search: PSNR = {psnr_full:.3f}, Runtime = {time_full:.3f} sec")
        print(
            f"Three-Step Search: PSNR = {psnr_three_step:.3f}, Runtime = {time_three_step:.3f} sec"
        )
        print("===================================================")
