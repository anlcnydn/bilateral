import numpy as np
from PIL import Image
import sys
import scipy.signal


def gaussian(x, sigma):
    return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(- (x ** 2) / (2 * sigma ** 2))


def bilateral_filter(source, radius, std_i, std_s):
    filtered_image = np.zeros_like(source).astype(float)
    W = 0

    pad = np.pad(source, (radius, radius), mode="symmetric")

    for i in range(-radius, radius):
        for j in range(-radius, radius):

            neighbour = pad[radius + i: radius + i + source.shape[0],
                            radius + j: radius + j + source.shape[1]]

            distance_x = (i)**2
            distance_y = (j)**2
            distance = np.sqrt(distance_x + distance_y)

            gi = gaussian((neighbour - source), std_i)
            gs = gaussian(distance, std_s)

            w = gi * gs
            W += w
            filtered_image += neighbour * w

    return np.around(filtered_image / W).astype(np.uint8)


if __name__ == "__main__":

    src = cv2.imread(str(sys.argv[1]), 0)
    dest = "filtered_image_own.png"

    with Image.open(src) as pic:

        pic = np.array(pic)

        filtered_image = bilateral_filter(pic, 5, 1, 3)

        with Image.fromarray(filtered_image) as output:
            output.save(dest, mode="L")
