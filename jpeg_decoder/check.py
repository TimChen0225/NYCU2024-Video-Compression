from struct import unpack


marker_mapping = {
    0xFFD8: "Start of Image",
    0xFFE0: "Application Default Header",
    0xFFDB: "Quantization Table",
    0xFFC2: "Start of Frame",
    0xFFC4: "Define Huffman Table",
    0xFFDA: "Start of Scan",
    0xFFD9: "End of Image",
}


class JPEG:
    def __init__(self, image_file):
        with open(image_file, "rb") as f:
            self.img_data = f.read()

    def decode(self):
        data = self.img_data
        while True:
            (marker,) = unpack(">H", data[0:2])
            print(marker_mapping.get(marker), hex(marker))
            if marker == 0xFFD8:  # soi
                data = data[2:]
            elif marker == 0xFFD9:  # eoi
                return
            elif marker == 0xFFDA:  # sos
                pass
                data = data[-2:]
            else:
                (lenchunk,) = unpack(">H", data[2:4])
                print("len:", lenchunk)
                chunk = data[4 : 2 + lenchunk]
                data = data[2 + lenchunk :]

                if marker == 0xFFC4:
                    pass
                elif marker == 0xFFDB:
                    pass
                elif marker == 0xFFC0:
                    pass

            if len(data) == 0:
                break


if __name__ == "__main__":
    img = JPEG("lena.jpg")
    img.decode()
