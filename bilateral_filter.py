import numpy as np
from PIL import Image
import sys
from skimage import color

class colors():
    
    def compute_LAB(self):
        try:
            self.LAB = color.rgb2lab(self.RGB/255.0)
        except:
            self.LAB = np.dstack((self.RGB/255.0, 
                        np.zeros_like(self.RGB), 
                        np.zeros_like(self.RGB)))
        
    def compute_RGB(self):
        self.RGB = color.lab2rgb(self.LAB) * 255.0
        self.RGB = self.RGB.astype(np.uint8)
          
    @property
    def R(self):
        return self.RGB[..., 0]
    
    @property
    def G(self):
        return self.RGB[..., 1]
    
    @property
    def B(self):
        return self.RGB[..., 2]
    
    @property
    def L(self):
        return self.LAB[..., 0]
    
    @property
    def A(self):
        return self.LAB[..., 1]
    
    @property
    def B(self):
        return self.LAB[..., 2]
    
    @R.setter
    def R(self, R):
        self.RGB[..., 0] = R
        
    @G.setter
    def G(self, G):
        self.RGB[..., 1] = G
    
    @B.setter
    def B(self, B):
        self.RGB[..., 2] = B
    
    @L.setter
    def L(self, L):
        self.LAB[..., 0] = L
        
    @A.setter
    def A(self, A):
        self.LAB[..., 1] = A
        
    @B.setter
    def B(self, B):
        self.LAB[..., 2] = B
        
    
    def __init__(self, picture):
        self.RGB = np.array(picture)
        
        try:
            if self.RGB.shape[2] != 3:
                self.RGB = np.dstack((self.RGB, self.RGB, self.RGB))
        except:
            self.RGB = np.dstack((self.RGB, self.RGB, self.RGB))
            
        self.compute_LAB()

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

    return filtered_image / W


if __name__ == "__main__":

    src = str(sys.argv[0])
    #src = "original_image_grayscale.png"
    dest = "filtered_image_own.png"

    with Image.open(src) as pic:

        pic = colors(pic)
        pic.L = bilateral_filter(pic.L, 4, 16.0, 12.0)
        pic.compute_RGB()

        with Image.fromarray(pic.RGB) as output:
            output.save(dest)
