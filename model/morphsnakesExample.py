import os
import logging
import cv2
import numpy as np
from imageio import imread
import matplotlib
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries, find_boundaries
from contour import count_population
import morphsnakes as ms
from threading import Thread


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


# in case you are running on machine without display, e.g. server
if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend.')
    # matplotlib.use('Agg')


PATH_IMG_LA = '../exampleImages/la_county.png'
PAUSE_TIME = 0.000001

num_districts = 8

# Load the image.
imgcolor = imread(PATH_IMG_LA)
imgcolor = imgcolor / 255
img = rgb2gray(imgcolor)

population_count = count_population(PATH_IMG_LA)
print(population_count)


def find_population_in_level_set(levelset):
    population = 0

    for i in range(len(levelset)):
        for j in range(len(levelset[i])):
            pix = (img[i][j])
            if (levelset[i][j] == 1) and pix < 0.9:
                # print(i, j, pix)
                population += 1

    # for level in levelset:
    #     for pixel in level:
    #         if pixel == 1:
    #             population += 1

    return population*200


def score(levelset, district_count, population):
    district_population = find_population_in_level_set(levelset)
    ideal_count = population / district_count

    ratio1 = district_population/ideal_count
    return ratio1


def visual_callback_2d(background, fig=None):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 2D images.

    Parameters
    ----------
    background : (M, N) array
        Image to be plotted as the background of the visual evolution.
    fig : matplotlib.figure.Figure
        Figure where results will be drawn. If not given, a new figure
        will be created.

    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.

    """

    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)

    plt.pause(PAUSE_TIME)

    def callback(levelset, counter, prevscore=0):
        # print(levelset.shape)
        if counter % 5 == 0:
            cur_score = score(levelset, num_districts, population_count)
            print('Districts: ', num_districts, ', Score:', cur_score)

            if (cur_score > 0.97):
                print('Done with contour, score=', cur_score)
                return cur_score

        if ax1.collections:
            del ax1.collections[0]

        ax1.contour(levelset, [0.5], colors='r')
        ax_u.set_data(levelset)
        fig.canvas.draw()

        plt.pause(PAUSE_TIME)
        return prevscore

    return callback


# def example_nodule():
#     logging.info('Running: example_nodule (MorphGAC)...')

#     # Load the image.
#     img = imread(PATH_IMG_NODULE)[..., 0] / 255.0

#     # g(I)
#     gimg = ms.inverse_gaussian_gradient(img, alpha=1000, sigma=5.48)

#     # Initialization of the level-set.
#     init_ls = ms.circle_level_set(img.shape, (100, 126), 20)

#     # Callback for visual plotting
#     callback = visual_callback_2d(img)

#     # MorphGAC.
#     ms.morphological_geodesic_active_contour(gimg, iterations=45,
#                                              init_level_set=init_ls,
#                                              smoothing=1, threshold=0.31,
#                                              balloon=1, iter_callback=callback)


# def example_starfish():
#     logging.info('Running: example_starfish (MorphGAC)...')

#     # Load the image.
#     imgcolor = imread(PATH_IMG_STARFISH) / 255.0
#     img = rgb2gray(imgcolor)

#     # g(I)
#     gimg = ms.inverse_gaussian_gradient(img, alpha=1000, sigma=2)

#     # Initialization of the level-set.
#     init_ls = ms.circle_level_set(img.shape, (163, 137), 135)

#     # Callback for visual plotting
#     callback = visual_callback_2d(imgcolor)

#     # MorphGAC.
#     ms.morphological_geodesic_active_contour(gimg, iterations=100,
#                                              init_level_set=init_ls,
#                                              smoothing=2, threshold=0.3,
#                                              balloon=-1, iter_callback=callback)


# def example_coins():
#     logging.info('Running: example_coins (MorphGAC)...')

#     # Load the image.
#     img = imread(PATH_IMG_COINS) / 255.0

#     # g(I)
#     gimg = ms.inverse_gaussian_gradient(img)

#     # Manual initialization of the level set
#     init_ls = np.zeros(img.shape, dtype=np.int8)
#     init_ls[10:-10, 10:-10] = 1

#     # Callback for visual plotting
#     callback = visual_callback_2d(img)

#     # MorphGAC.
#     ms.morphological_geodesic_active_contour(gimg, 230, init_ls,
#                                              smoothing=1, threshold=0.69,
#                                              balloon=-1, iter_callback=callback)

def example_la():
    logging.info('Running: example_lakes (MorphACWE)...')
    global img
    global num_districts
    # mouse callback function
    points = []

    def note_point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            points.append((y, x))

    # Create a black image, a window and bind the function to window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', note_point)

    while(1):
        cv2.imshow('image', imgcolor)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('p'):
            print(points)
        elif k == ord('q'):
            break
    cv2.destroyAllWindows()

    num_districts = len(points)

    # MorphACWE does not need g(I)
    all_scores = []
    # Initialization of the level-set.
    for point in points:
        init_ls = ms.circle_level_set(img.shape, point, 40)

        # Callback for visual plotting
        callback = visual_callback_2d(imgcolor)

        # Morphological Chan-Vese (or ACWE)
        r, score = ms.morphological_chan_vese(img, iterations=100,
                                              init_level_set=init_ls,
                                              smoothing=3, lambda1=1, lambda2=1,
                                              iter_callback=callback)

        all_scores.append(score)
        bounder = find_boundaries(r, mode='thick').astype(np.uint8)
        imgcolor[bounder != 0] = (255, 0, 0, 1)

        r = 1 - r
        imgmod = cv2.bitwise_and(imgcolor, imgcolor, mask=r)
        img = rgb2gray(imgmod)

    print('Total Score:', all_scores)
    total_score = 0
    for score in all_scores:
        if score > 1:
            total_score += (1 - (score - 1))
        else:
            total_score += score

    print(total_score)

    # mouse callback function
    # points = []

    # def note_point(event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDBLCLK:
    #         points.append((y, x))

    # # Create a black image, a window and bind the function to window
    # cv2.namedWindow('image')
    # cv2.setMouseCallback('image', note_point)

    # while(1):
    #     cv2.imshow('image', imgcolor)
    #     k = cv2.waitKey(20) & 0xFF
    #     if k == ord('p'):
    #         print(points)
    #     elif k == ord('q'):
    #         break
    # cv2.destroyAllWindows()

    # # MorphACWE does not need g(I)

    # # Initialization of the level-set.
    # init_ls = ms.circle_level_set(img.shape, points[0], 40)

    # # Callback for visual plotting
    # callback = visual_callback_2d(imgcolor)

    # # Morphological Chan-Vese (or ACWE)
    # Thread(target=ms.morphological_chan_vese(img, iterations=300,
    #                                          init_level_set=init_ls,
    #                                          smoothing=3, lambda1=1, lambda2=1,
    #                                          iter_callback=callback)).start()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    example_la()

    # Uncomment the following line to see a 3D example
    # This is skipped by default since mplot3d is VERY slow plotting 3d meshes
    # example_confocal3d()

    logging.info("Done.")
    plt.show()
