import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops, label
from pathlib import Path

def count_vlines(region):
    return np.all(region.image, axis = 0).sum()
def count_glines(region):
    return np.all(region.image, axis = 0).sum()

def extractor(region):
    area = region.area / region.image.size

    cx, cy = region.centroid_local
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]

    perimeter = region.perimeter / region.image.size
    eccen = region.eccentricity
    vertical = np.all(region.image, axis=0).sum()
    gorizontal = np.all(region.image, axis=1).sum()
    euler = 1 - region.euler_number
    solidity = region.solidity

    h = region.image.shape[0]
    w = region.image.shape[1]
    width_to_height = w / h

    cy_new = int(region.centroid_local[0])
    cx_new = int(region.centroid_local[1])
    row_cross = np.sum(region.image[cy_new, :-1] != region.image[cy_new, 1:])
    col_cross = np.sum(region.image[:-1, cx_new] != region.image[1:, cx_new])

    return np.array([
        area, cy, cx, perimeter, eccen, euler, solidity, width_to_height, row_cross, col_cross
    ])

def norm_l1(v1, v2):
    return ((v1 - v2) ** 2).sum() ** 0.5

def classificator(v, templates):
    result = "_"
    min_dist = 10 ** 16
    for key in templates:
        d = norm_l1(v, templates[key])
        if d < min_dist:
            result = key
            min_dist = d
    return result

alphabet = plt.imread("alphabet.png")[:, :, :-1]

gray = alphabet.mean(axis=2)
binary = gray > 0
labeled = label(binary)
regions = regionprops(labeled)

symbols = plt.imread("alphabet-small.png")[:, :, :-1]
gray = symbols.mean(axis=2)
binary = gray < 1
slabeled = label(binary)
sregions = regionprops(slabeled)


templates = {"A": extractor(sregions[2]),
             "B": extractor(sregions[3]), 
             "8": extractor(sregions[0]), 
             "0": extractor(sregions[1]), 
             "1": extractor(sregions[4]), 
             "W": extractor(sregions[5]), 
             "X": extractor(sregions[6]), 
             "*": extractor(sregions[7]),
             "-": extractor(sregions[9]), 
             "/": extractor(sregions[8])}

# print(templates)
# for i, region in enumerate(sregions):
#     v = extractor(region)
#     plt.subplot(2, 5, i+1)
#     plt.title(classificator(v, templates))
#     plt.imshow(region.image)

result = {}
for region in regions:
    v = extractor(region)
    symbol = classificator(v, templates)
    result[symbol] = result.get(symbol, 0) + 1

print(result)