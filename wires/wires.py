import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.morphology import binary_closing, binary_opening, binary_erosion, binary_dilation

for file_num in range(1,7):
    file_name = f"wires{file_num}npy.txt"
    data = np.load(file_name)
    labeled = label(data)
    print(f"Файл {file_name}")
    for i in range(1, np.max(labeled)+1):
        result = binary_erosion(labeled == i)
        result = label(result)
        cnt = np.max(result)
        if cnt > 1:
            print(f"Провод {i} разделен на {np.max(result)}")
        elif cnt == 1:
            print(f"Провод {i} не разделён")
        else:
            print(f"Провод {i} не существует")        


plt.imshow(labeled)
plt.show()