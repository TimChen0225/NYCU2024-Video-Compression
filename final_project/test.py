from PIL import Image

image = Image.open("simple_version.jpg")
print(image.quantization)  # 查看實際量化表
