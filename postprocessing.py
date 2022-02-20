import cv2
import numpy as np
from PIL import Image


class PostProcess():
    def __init__(self):
        self.bg = cv2.imread("./background/bg.jpg")


    def postprocess(self, mask, w, h):
        
        bg = cv2.resize(self.bg, (w,h))
        background = Image.fromarray(bg)
        foreground = Image.fromarray(mask)

        background.paste(foreground, (0,0), foreground)
        
        # output_image = cv2.addWeighted(bg, 0.4, mask, 0.1, 0)
        return np.asarray(background)

    
