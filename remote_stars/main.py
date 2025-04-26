import socket
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

host="84.237.21.36"
port = 5152

def distanse(point1, point2):
    return ((point2[0] - point1[0])**2 + (point2[1]-point1[1])**2 )**0.5

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((host, port))
    beat = b"nope"

    plt.ion()
    plt.figure()

    for i in range(10):
        sock.send(b"get")
        bts = recvall(sock, 40002)
            
        im1 = np.frombuffer(bts[2:], dtype="uint8").reshape(bts[0],bts[1])
        binary = im1 > 0
        labeled = label(binary)
        regions = regionprops(labeled)
        centers = []
        for region in regions:
            centers.append(region.centroid)
        result = distanse(centers[0],centers[1])
        sock.send(f"{result:.1f}".encode())

        plt.imshow(im1)
        plt.pause(1)

        sock.send(b"beat")
        beat = sock.recv(10)
        print(result, beat)
        