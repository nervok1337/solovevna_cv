import numpy as np
import matplotlib.pyplot as plt

def match(a, masks):
    for mask in masks:
        if np.all(a == mask):
            return True
    return False


def count_objects(image):
    E = 0
    for y in range(0, image.shape[0] - 1):
        for x in range(0, image.shape[1] - 1):
            sub = image[y : y + 2, x : x + 2]
            if match(sub, external):
                E += 1
            elif match(sub, internal):
                E -= 1
            elif match(sub, cross):
                E += 2
    return E / 4

external = np.array([
    [[0,0],[0,1]],
    [[0,0],[1,0]],
    [[0,1],[0,0]],
    [[1,0],[0,0]]
])
internal = np.logical_not(external)
cross = np.array([
    [[1,0],[0,1]],
    [[0,1],[1,0]]
])

image1 = np.load("example1.npy")
image1[image1 > 0] = 1
print(count_objects(image1))

image2 = np.load("example2.npy")
image2[image2 > 0] = 1
print(sum([count_objects(image2[:,:,i]) for i in range(image2.shape[2])]))

#plt.imshow(image)
#plt.show()