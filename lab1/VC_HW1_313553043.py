import cv2
import numpy as np

img = cv2.imread("lena.png")

if img is None:
    print("image read fail")
    exit()


b = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
g = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
r = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
y = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
u = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
v = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
cb = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
cr = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)


b[:, :] = img[:, :, 0]
g[:, :] = img[:, :, 1]
r[:, :] = img[:, :, 2]

y[:, :] = 0.299 * r[:, :] + 0.587 * g[:, :] + 0.114 * b[:, :]
u[:, :] = -0.169 * r[:, :] - 0.331 * g[:, :] + 0.5 * b[:, :] + 128
v[:, :] = 0.5 * r[:, :] - 0.419 * g[:, :] - 0.081 * b[:, :] + 128

cb[:, :] = -0.168736 * r[:, :] - 0.331264 * g[:, :] + 0.5 * b[:, :] + 128
cr[:, :][:, :] = 0.5 * r - 0.418688 * g[:, :] - 0.081312 * b[:, :] + 128

# cv2.imshow('lena',img)
# cv2.waitKey(0)

cv2.imshow("r.png", r)
cv2.imshow("g.png", g)
cv2.imshow("b.png", b)
cv2.imshow("y.png", y)
cv2.imshow("u.png", u)
cv2.imshow("v.png", v)
cv2.imshow("cb.png", cb)
cv2.imshow("cr.png", cr)
cv2.waitKey(0)


# while True:
#     key = input("store images?(Y/N): ").strip().upper()
#     if key == "Y":
#         cv2.imwrite("r.png", r)
#         cv2.imwrite("g.png", g)
#         cv2.imwrite("b.png", b)
#         cv2.imwrite("y.png", y)
#         cv2.imwrite("u.png", u)
#         cv2.imwrite("v.png", v)
#         cv2.imwrite("cb.png", cb)
#         cv2.imwrite("cr.png", cr)
#         print("image saved")
#         break
#     elif key == "N":
#         break
#     else:
#         print("Invalid input")
