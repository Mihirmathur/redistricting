import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans
import cv2

def find_centers(PATH, viz=False):

    def getColor(cluster):
        if cluster == 0:
            return (0, 123, 0)
        if cluster == 1:
            return (0, 0, 123)
        if cluster == 2:
            return (123, 0, 0)
        if cluster == 3:
            return (220, 0, 123)
        if cluster == 4:
            return (123, 220, 0)
        if cluster == 5:
            return (50, 0, 123)
        if cluster == 6:
            return (20, 100, 50)
        else:
            return (23, 153, 123)


    n = 8
    img = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    pts = np.array([[0, 0]])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i][j] < 250):
                pts = np.append(pts, [[i, j]], axis=0)

    clf = KMeans(n_clusters=n, random_state=0)
    clf.fit(pts)
    
    if viz:
        clf.fit(pts)
        print('Num clusters=', n)
        print('Cluster centers:', clf.cluster_centers_)

        img = cv2.imread(PATH)
        for p, l in zip(pts, clf.labels_):
            img[p[0]][p[1]] = getColor(l)
            
        cv2.imwrite('result_KMeans.jpg', img)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])
        plt.show()
    
    return clf.cluster_centers_
