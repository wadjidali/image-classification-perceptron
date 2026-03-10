import os
from preprocessing import transform_image

def load_dataset(path):
    data = []
    labels = []
    
    for label in os.listdir(path):
        folder = os.path.join(path, label)
        
        for img in os.listdir(folder):
            image = transform_image(os.path.join(folder, img))
            data.append(image.flatten())
            labels.append(label)
    
    return data, labels
