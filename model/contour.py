import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import io
from skimage.filters import gaussian
from skimage.segmentation import active_contour, quickshift
from skimage.util import img_as_float
from random import randint
import cv2
from shapely.geometry import Polygon
from shapely.geometry import Point


img = io.imread('../viz_population/la_county.png')



radius = 100
centers = [(570, 290, 100)]

for i in range(5):
    x = randint(0, 600)
    y = randint(0, 600)
    r = randint(30, 100)
    centers.append((x, y, r))

init_array = []
s = np.linspace(0, 2*np.pi, 400)

for center in centers:
    r = center[0] + center[2]*np.sin(s)
    c = center[1] + center[2]*np.cos(s)
    init = np.array([r, c]).T
    init_array.append(init)


fig, ax = plt.subplots(figsize=(7, 7))
snakes = []
for i in init_array:
    snake = active_contour(img,
                           i, alpha=0.1, beta=10, gamma=0.001,
                           coordinates='rc')
    snakes.append(snake)
    ax.plot(i[:, 1], i[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)

# print(snakes[0])
ax.imshow(img, cmap=plt.cm.gray)

ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
# cv2.fillPoly(img, pts=[snakes[0]], color=(15, 255, 255))
# cv2.imshow("", img)


def count_population(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    n_white_pix = np.sum(img != 255) 
    return n_white_pix * 100




def find_population_in_contour (snake):

    pl = Polygon(snake)

    minx, miny, maxx, maxy = pl.bounds
    minx, miny, maxx, maxy = int(minx), int(miny), int(maxx), int(maxy)
    box_patch = [[x,y] for x in range(minx,maxx+1) for y in range(miny,maxy+1)]
    pixels = []
    for pb in box_patch: 
      pt = Point(pb[0],pb[1])
      if(pl.contains(pt)):
        pixels.append([int(pb[0]), int(pb[1])])

    xcord, ycord = zip(*pixels)

    newl1 = []
    newl2 = []

    for p in pixels :
        x = p[0]
        y = p[1]

        pix = (img[x,y][:-1])

        if (pix != [255,255,255]).all() :
            newl1.append(x)
            newl2.append(y)

    return (newl1, newl2)



def score(snake, district_count):
    (l1, l2) = find_population_in_contour(snake)
    snake_count = len(l1)
    population_count = count_population('../viz_population/la_county.png')
    ideal_count = population_count / district_count

    ratio1 = snake_count/ideal_count
    ratio2 = ideal_count/snake_count

    if (ratio1 == 1):
        return ratio1

    else:
        return min(ratio1, ratio2)


print(score(snakes[0], 8))


# print('population count')
# print(count_population('../viz_population/la_county.png'))

# newl1 = find_population_in_contour(snakes)[0]
# newl2 = find_population_in_contour(snakes)[1]

# ax.plot(newl2, newl1, 'b')

# plt.show()




