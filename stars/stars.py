import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.morphology import binary_closing, binary_opening, binary_erosion, binary_dilation

data = np.load("stars.npy")
data = data.astype(int)

erosion_data = binary_erosion(data)
labeling = label(data)
gen_cnt = np.max(labeling)

cnt = 0

for y in range(0, erosion_data.shape[0] - 1):
    for x in range(0, erosion_data.shape[1] - 1):
        part = erosion_data[y, x:x+2]
        if sum(part) == 2:
            cnt+=1

print(gen_cnt - cnt)

plt.imshow(labeling)
plt.show()