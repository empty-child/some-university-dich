import binascii
import collections
import time
from bs4 import BeautifulSoup
from heapq import heappush, heappop, heapify

with open("2.jpg", "rb") as f:
    data = f.read()
data = binascii.hexlify(data)
data = data.decode('ascii')
print("Input sequence: " + data)


def rle():
    t1 = time.clock()
    i, j = 0, 1
    prev = ""
    output = ""
    while i < len(data):
        if data[i] != prev:
            if i >= 1:
                output += str(j)
            output += data[i]
            prev = data[i]
            j = 1
        else:
            j += 1
        i += 1
    if j >= 1:
        output += str(j)
    t2 = time.clock()
    print("RLE output sequence: " + output)
    return t2-t1


def huffman():
    t1 = time.clock()

    def encode(symb2freq):
        heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
        heapify(heap)
        while len(heap) > 1:
            lo = heappop(heap)
            hi = heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        return sorted(heappop(heap)[1:])
    cnt = collections.Counter(data)
    huffman_codes = dict(encode(cnt))
    output = ""
    for i in data:
        output += huffman_codes[i]
    t2 = time.clock()
    print("Huffman output sequence: " + output)
    return t2-t1


def lzw():
    t1 = time.clock()
    charlist = []
    output = []
    for i in data:
        if i not in charlist:
            charlist.append(i)
    charlist.sort()
    w = ""
    for i in data:
        wi = w + i
        if wi in charlist:
            w = wi
        else:
            charlist.append(wi)
            output.append(charlist.index(w))
            w = i
    if w:
        output.append(charlist.index(w))
    print("LZW output sequence: " + str(output))
    t2 = time.clock()
    return t2-t1

tr = round(rle(),4)
th = round(huffman(),4)
tl = round(lzw(),4)

with open("out.html", "rb") as f:
    html = f.read()

soup = BeautifulSoup(html, "lxml")
for el in [["rle", tr],["hfm", th],["lzw", tl]]:
    elem = soup.find(attrs={"id": el[0]})
    elem.contents.clear()
    elem.insert(0, str(el[1]))

with open("out.html", "wb") as f:
    html = soup.encode("utf-8")
    f.write(html)

# with open("1.dat", "wb") as f:
#     data = out.encode("utf-8")
#     data = binascii.unhexlify(data)
#     # print(data)
#     f.write(out) # не преобразует хаффмана, надо доделать