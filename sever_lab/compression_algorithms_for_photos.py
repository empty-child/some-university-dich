import binascii

f = open("1.jpg", "rb")
data = f.read()
f.close()
data = binascii.hexlify(data)
data = data.decode('ascii')
print(data)


def rle(data):
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
    return output

out = rle(data)
print(out)

f = open("1.dat", "wb")
data = out.encode("ascii")
data = binascii.unhexlify(data)
print(data)
f.write(data)
f.close()