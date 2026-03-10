from PIL import Image
import numpy as np

def transform_image(path):
    img = Image.open(path)
    img = img.convert('L')
    img = img.resize((64,64))
    return np.array(img)/255
