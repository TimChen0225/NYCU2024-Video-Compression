from struct import *
import matplotlib.pyplot as plt
import numpy as np

marker_mapping = {
    0xFFD8: "Start of Image",
    0xFFE0: "Application Default Header",
    0xFFDB: "Quantization Table",
    0xFFC0: "Start of Frame",
    0xFFC4: "Define Huffman Table",
    0xFFDA: "Start of Scan",
    0xFFD9: "End of Image",
}

subsample_mapping = {"11": "4:4:4", "21": "4:2:2", "41": "4:1:1", "22": "4:2:0"}


def GetArray(type, l, length):
    s = ""
    for i in range(length):
        s = s + type
    return list(unpack(s, l[:length]))


def DecodeNumber(code, bits):
    l = 2 ** (code - 1)
    if bits >= l:
        return bits
    else:
        return bits - (2 * l - 1)


class JPEG:
    def __init__(self, image_file):
        with open(image_file, "rb") as f:
            self.img_data = f.read()
        self.quant = {}
        self.huffman_tables = {}
        self.height = 0
        self.width = 0
        self.decoded_data = []
        self.horizontalFactor = []
        self.verticalFactor = []
        self.quantMapping = []
        self.components = {}

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
                self.decodeSOS(data[2:-2])
                data = data[-2:]
            else:
                (lenchunk,) = unpack(">H", data[2:4])
                print("len:", lenchunk)
                chunk = data[4 : 2 + lenchunk]
                data = data[2 + lenchunk :]

                if marker == 0xFFC4:
                    self.decodeHuffmanTable(chunk)
                elif marker == 0xFFDB:
                    self.DefineQuantizationTables(chunk)
                elif marker == 0xFFC0:
                    self.decodeFrameHeader(chunk)

            if len(data) == 0:
                break

    def DefineQuantizationTables(self, data):
        offset = 0
        while offset < len(data):
            (id,) = unpack("B", data[offset : offset + 1])
            self.quant[id] = GetArray("B", data[offset + 1 : offset + 65], 64)
            print("#{} Table".format(id))
            print("Elements: ", self.quant[id])
            offset += 65

    def decodeFrameHeader(self, data):
        # BaselineDCT
        precison, self.height, self.width, self.components = unpack(">BHHB", data[0:6])
        for i in range(self.components):
            id, factor, QtbId = unpack("BBB", data[6 + i * 3 : 9 + i * 3])
            h, v = (factor >> 4, factor & 0x0F)
            self.horizontalFactor.append(h)
            self.verticalFactor.append(v)
            self.quantMapping.append(QtbId)

        print("size {}x{}".format(self.width, self.height))
        print(
            "subsampling {}".format(
                subsample_mapping.get(
                    "{}{}".format(
                        int(max(self.horizontalFactor) / self.horizontalFactor[0]),
                        int(max(self.verticalFactor) / self.verticalFactor[0]),
                    )
                )
            )
        )

    def decodeHuffmanTable(self, data):
        offset = 0
        tbcount = 0
        print(len(data))
        while offset < len(data):
            (tcth,) = unpack("B", data[offset : offset + 1])
            tc, th = (tcth >> 4, tcth & 0x0F)
            offset += 1

            # Extract the 16 bytes containing length data
            lengths = unpack("BBBBBBBBBBBBBBBB", data[offset : offset + 16])
            offset += 16

            # Extract the elements after the initial 16 bytes
            elements = []
            for i in lengths:
                elements += unpack("B" * i, data[offset : offset + i])
                offset += i

            print(
                "#{}: {} {} Table".format(
                    tbcount, "Luma" if th == 0 else "Chroma", "DC" if tc == 0 else "AC"
                )
            )
            print("lengths: ", lengths)
            print("Elements: ", elements)

            # tc = 0(DC), 1(AC)
            # th = 0(Luminance), 1(Chroma)
            hf = HuffmanTable()
            hf.GetHuffmanBits(lengths, elements)
            self.huffman_tables[tc << 1 | th] = hf
            tbcount += 1

    def decodeSOS(self, data):
        # BaselineDCT
        ls, ns = unpack(">HB", data[0:3])
        # print("ns=", ns)
        csj = unpack("BB" * ns, data[3 : 3 + 2 * ns])
        dcTableMapping = []
        acTableMapping = []
        for i in range(ns):
            dcTableMapping.append(csj[i * 2 + 1] >> 4)
            acTableMapping.append(csj[i * 2 + 1] & 0x0F)
        data = data[6 + 2 * ns :]

        # Replace 0xFF00 with 0xFF
        i = 0
        while i < len(data) - 1:
            (m,) = unpack(">H", data[i : i + 2])
            if m == 0xFF00:
                data = data[: i + 1] + data[i + 2 :]
            i = i + 1

        img = Image(self.height, self.width)
        st = Stream(data)
        oldlumdccoeff, oldCbdccoeff, oldCrdccoeff = 0, 0, 0
        for y in range(self.height // 8):
            for x in range(self.width // 8):
                matL, oldlumdccoeff = self.BuildMatrix(
                    st,
                    dcTableMapping[0],
                    acTableMapping[0],
                    self.quant[self.quantMapping[0]],
                    oldlumdccoeff,
                )
                matCb, oldCrdccoeff = self.BuildMatrix(
                    st,
                    dcTableMapping[1],
                    acTableMapping[1],
                    self.quant[self.quantMapping[1]],
                    oldCrdccoeff,
                )
                matCr, oldCbdccoeff = self.BuildMatrix(
                    st,
                    dcTableMapping[2],
                    acTableMapping[2],
                    self.quant[self.quantMapping[2]],
                    oldCbdccoeff,
                )
                img.DrawMatrix(y, x, matL.base, matCb.base, matCr.base)

        img.ycbcr2rgb()
        self.decoded_data = img.img

    def BuildMatrix(self, st, dcTableId, acTableId, quant, olddccoeff):
        i = DCT()

        code = self.huffman_tables[0b00 | dcTableId].GetCode(st)
        bits = st.GetBitN(code)
        dccoeff = DecodeNumber(code, bits) + olddccoeff
        i.base[0] = dccoeff

        l = 1
        while l < 64:
            code = self.huffman_tables[0b10 | acTableId].GetCode(st)
            if code == 0:
                # EOB
                break

            if code == 0xF0:
                # ZRL
                l += 16
                continue
            elif code > 15:
                l += code >> 4
                code = code & 0x0F

            bits = st.GetBitN(code)

            if l < 64:
                coeff = DecodeNumber(code, bits)
                i.base[l] = coeff
                l += 1

        i.base = np.multiply(i.base, quant)
        i.rearrange_using_zigzag()
        i.perform_IDCT()

        return i, dccoeff


class HuffmanTable:
    def __init__(self):
        self.root = []
        self.elements = []

    def BitsFromLengths(self, root, element, pos):
        if isinstance(root, list):
            if pos == 0:
                if len(root) < 2:
                    root.append(element)
                    return True
                return False
            for i in [0, 1]:
                if len(root) == i:
                    root.append([])
                if self.BitsFromLengths(root[i], element, pos - 1) == True:
                    return True
        return False

    def GetHuffmanBits(self, lengths, elements):
        self.elements = elements
        ii = 0
        for i in range(len(lengths)):
            for j in range(lengths[i]):
                self.BitsFromLengths(self.root, elements[ii], i)
                ii += 1

    def Find(self, st):
        r = self.root
        while isinstance(r, list):
            r = r[st.GetBit()]
        return r

    def GetCode(self, st):
        while True:
            res = self.Find(st)
            if res == 0:
                return 0
            elif res != -1:
                return res


class Stream:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def GetBit(self):
        b = self.data[self.pos >> 3]
        s = 7 - (self.pos & 0x7)
        self.pos += 1
        return (b >> s) & 1

    def GetBitN(self, l):
        val = 0
        for i in range(l):
            val = val * 2 + self.GetBit()
        return val


class Image:
    def __init__(self, height, width):
        self.width = width
        self.height = height
        # print("!!! image height:", height, "width:", width)
        # [0] = Y, [1] = Cb, [2]=Cr
        self.img = []
        for i in range(3):
            self.img.append(np.zeros((self.height, self.width)))

    def ycbcr2rgb(self):
        # Level-shifting
        for i in range(3):
            self.img[i] += 128

        # Merge multiple 2D arrays into a 3D array
        self.img = np.dstack(tuple([i for i in self.img]))

        # Convert image from YCbCr to RGB
        self.img[:, :, 1:] -= 128
        m = np.array(
            [
                [1.000, 1.000, 1.000],
                [0.000, -0.344136, 1.772],
                [1.402, -0.714136, 0.000],
            ]
        )
        rgb = np.dot(self.img, m)
        self.img = np.clip(rgb, 0, 255)

        # Convert a 3D array to multiple 2D arrays
        self.img = [self.img[:, :, i] for i in range(3)]

    def DrawMatrix(self, y, x, L, Cb, Cr):
        _x = x * 8
        _y = y * 8
        self.img[0][_y : _y + 8, _x : _x + 8] = L.reshape((8, 8))
        self.img[1][_y : _y + 8, _x : _x + 8] = Cb.reshape((8, 8))
        self.img[2][_y : _y + 8, _x : _x + 8] = Cr.reshape((8, 8))


class DCT:
    def __init__(self):
        self.base = np.zeros(64)
        self.zigzag = np.array(
            [
                [0, 1, 5, 6, 14, 15, 27, 28],
                [2, 4, 7, 13, 16, 26, 29, 42],
                [3, 8, 12, 17, 25, 30, 41, 43],
                [9, 11, 18, 24, 31, 40, 44, 53],
                [10, 19, 23, 32, 39, 45, 52, 54],
                [20, 22, 33, 38, 46, 51, 55, 60],
                [21, 34, 37, 47, 50, 56, 59, 61],
                [35, 36, 48, 49, 57, 58, 62, 63],
            ]
        ).flatten()
        # Generate 2D-DCT matrix
        L = 8
        C = np.zeros((L, L))
        for k in range(L):
            for n in range(L):
                C[k, n] = np.sqrt(1 / L) * np.cos(np.pi * k * (1 / 2 + n) / L)
                if k != 0:
                    C[k, n] *= np.sqrt(2)
        self.dct = C

    def perform_DCT(self):
        self.base = np.kron(self.dct, self.dct) @ self.base

    def perform_IDCT(self):
        self.base = np.kron(self.dct.transpose(), self.dct.transpose()) @ self.base

    def rearrange_using_zigzag(self):
        newidx = np.ones(64).astype("int8")
        for i in range(64):
            newidx[list(self.zigzag).index(i)] = i
        self.base = self.base[newidx]


if __name__ == "__main__":
    jpeg = JPEG("test.jpg")
    jpeg.decode()
    img = np.dstack(tuple([i for i in jpeg.decoded_data])).astype(np.uint8)

    img = plt.imshow(img, interpolation="nearest")
    plt.axis("off")
    plt.show()
    plt.savefig("output.png", bbox_inches="tight")
