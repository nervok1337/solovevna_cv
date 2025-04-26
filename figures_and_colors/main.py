import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv
from collections import defaultdict

def replace_colors(hues, threshold=0.005):
    sorted_hues = np.sort(hues)
    result = []
    group_min = sorted_hues[0]

    result.append(group_min)

    for i in range(1, len(sorted_hues)):
        if abs(sorted_hues[i] - group_min) <= threshold:
            result.append(group_min)          
        else:
            group_min = sorted_hues[i]
            result.append(group_min)

    return np.array(result)

def hues_count(colors):
    cnt_hues = defaultdict(int)
    for value in colors:
        cnt_hues[str(value)] += 1

    for value, count in cnt_hues.items():
        print(f"\tОттенок {value}: {count} ")

image = plt.imread("balls_and_rects.png")
gray = image.mean(axis=2)
binary = gray > 0
labeled = label(binary)
regions = regionprops(labeled)

hues = defaultdict(int)
cnt_balls = 0
cnt_rectangles = 0
colors_balls = []
colors_rectangles = []

for region in regions:
    h, w = region.image.shape
    area = h * w
  
    ratio = region.minor_axis_length / region.major_axis_length

    y, x = region.centroid
    hue = round(rgb2hsv(image[int(y), int(x)])[0], 3)
 
    if region.area != area and ratio > 0.9:
        cnt_balls += 1
        colors_balls.append(hue)
    else:
        cnt_rectangles += 1
        colors_rectangles.append(hue)


print(f'Количество кругов: {cnt_balls}')
print(f'Количество прямоугольников: {cnt_rectangles}')
print(f'Количество всех фигур: {np.max(labeled)}')

colors_balls = replace_colors(colors_balls)
colors_rectangles = replace_colors(colors_rectangles)

print("\nКруги:")
hues_count(colors_balls)
print("\nПрямоугольники:")
hues_count(colors_rectangles)



