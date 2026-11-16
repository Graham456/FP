"""32-bit adaptive binary arithmetic coding (not CABAC).

Eight independent probability models, one per bit position in each input byte.
Counts start at (1,1), halve at 16384, and reset for every frame packet.
E1/E2 normalization emits bits; E3 normalization defers underflow bits.
"""
TOP = (1 << 32) - 1
HALF, QUARTER = 1 << 31, 1 << 30


class BitsOut:
    def __init__(self):
        self.data = bytearray()
        self.value = self.count = 0

    def put(self, bit):
        self.value = (self.value << 1) | bit
        self.count += 1
        if self.count == 8:
            self.data.append(self.value)
            self.value = self.count = 0

    def finish(self):
        if self.count:
            self.data.append(self.value << (8 - self.count))
        return bytes(self.data)


def update(counts, context, bit):
    counts[context][bit] += 1
    if sum(counts[context]) >= 16384:
        counts[context] = [(v + 1) // 2 for v in counts[context]]


def compress(data):
    low, high, pending = 0, TOP, 0
    counts = [[1, 1] for _ in range(8)]
    out = BitsOut()

    def emit(bit, pending):
        out.put(bit)
        for _ in range(pending):
            out.put(1 - bit)

    for byte in data:
        for context in range(8):
            bit = (byte >> (7 - context)) & 1
            zero, one = counts[context]
            split = low + (high - low + 1) * zero // (zero + one)
            if bit:
                low = split
            else:
                high = split - 1
            while True:
                if high < HALF:
                    emit(0, pending)
                    pending = 0
                elif low >= HALF:
                    emit(1, pending)
                    pending = 0
                    low -= HALF
                    high -= HALF
                elif low >= QUARTER and high < 3 * QUARTER:
                    pending += 1
                    low -= QUARTER
                    high -= QUARTER
                else:
                    break
                low *= 2
                high = high * 2 + 1
            update(counts, context, bit)
    emit(0 if low < QUARTER else 1, pending + 1)
    return out.finish()


def decompress(data, size):
    if not data or size < 0:
        raise ValueError("Invalid arithmetic payload")
    pos = 0

    def get():
        nonlocal pos
        bit = (data[pos // 8] >> (7 - pos % 8)) & 1 if pos < len(data) * 8 else 0
        pos += 1
        return bit

    low, high, code = 0, TOP, 0
    for _ in range(32):
        code = code * 2 + get()
    counts = [[1, 1] for _ in range(8)]
    out = bytearray()
    for _ in range(size):
        byte = 0
        for context in range(8):
            zero, one = counts[context]
            split = low + (high - low + 1) * zero // (zero + one)
            bit = int(code >= split)
            if bit:
                low = split
            else:
                high = split - 1
            byte = byte * 2 + bit
            while True:
                if high < HALF:
                    pass
                elif low >= HALF:
                    low -= HALF
                    high -= HALF
                    code -= HALF
                elif low >= QUARTER and high < 3 * QUARTER:
                    low -= QUARTER
                    high -= QUARTER
                    code -= QUARTER
                else:
                    break
                low *= 2
                high = high * 2 + 1
                code = code * 2 + get()
            update(counts, context, bit)
        out.append(byte)
    return bytes(out)
