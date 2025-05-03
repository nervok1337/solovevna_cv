import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_erosion, binary_dilation
from scipy.ndimage import binary_fill_holes

all_pencils = 0
for i in range(1, 13):
    image = plt.imread(f"./images/img ({i}).jpg")
    image = image[50:-50, 50:-50]
    gray = image.mean(axis=2)

    binary = gray < 120
    binary_close = binary_closing(binary, np.ones((35,35)))
    binary_fill = binary_fill_holes(binary_close)
    binary_erose = binary_erosion(binary_fill, np.ones((5,5)))
    binary_dil = binary_dilation(binary_erose, np.ones((5,5)))

    labeled = label(binary_dil)
    regions = regionprops(labeled)

    curr_pencils = 0
    for region in regions:
        if 1 > region.eccentricity > 0.99:
            all_pencils += 1
            curr_pencils += 1
    print(f"Изображение {i}: {curr_pencils}")
print(f"Всего: {all_pencils}")