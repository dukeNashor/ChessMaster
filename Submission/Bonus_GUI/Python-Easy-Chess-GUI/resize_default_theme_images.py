from skimage import io
from skimage.transform import resize

import os

IMAGE_PATH = 'Images/60'  # path to the chess pieces
SCALED_IMAGE_PATH = 'Images/60_scaled'  # path to the chess pieces

os.makedirs(SCALED_IMAGE_PATH, exist_ok = True)

names = os.listdir(IMAGE_PATH)

for name in names:
    img = io.imread(IMAGE_PATH + "/" + name)
    img_resized = resize(img, (50, 50), anti_aliasing = True)
    io.imsave(SCALED_IMAGE_PATH + "/" + name, img_resized)