import torch

# 檢查 CUDA 是否可用
print("CUDA 是否可用:", torch.cuda.is_available())

# 列出 GPU 名稱
if torch.cuda.is_available():
    print("GPU 名稱:", torch.cuda.get_device_name(0))
    print("CUDA 版本:", torch.version.cuda)
else:
    print("無法使用 GPU，請檢查安裝設定。")
