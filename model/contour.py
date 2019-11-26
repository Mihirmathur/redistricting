import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import io
from skimage.filters import gaussian
from skimage.segmentation import active_contour, quickshift
from skimage.util import img_as_float
from random import randint
import cv2

img = io.imread('../viz_population/la_county.png')
# img = rgb2gray(img)

radius = 100
centers = [(570, 290, 100), (253, 405, 100)]

# for i in range(5):
#     x = randint(0, 600)
#     y = randint(0, 600)
#     r = randint(30, 100)
#     centers.append((x, y, r))

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
    snake = active_contour(img, i, alpha=0.01, beta=0.01,
                           gamma=0.001, w_line=-0.75,
                           boundary_condition='periodic',
                           max_iterations=5000,
                           w_edge=1, coordinates='rc', convergence=0.01)
    snakes.append(snake)
    ax.plot(i[:, 1], i[:, 0], '--r', lw=2)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=2)
    ax.plot()

# print(snakes[0])
ax.imshow(img, cmap=plt.cm.gray)

ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
# cv2.fillPoly(img, pts=[snakes[0]], color=(15, 255, 255))
# cv2.imshow("", img)
plt.show()
