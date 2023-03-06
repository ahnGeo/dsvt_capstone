import numpy
import PIL
import pickle

with open("data.pkl", 'rb') as f :
    data = pickle.load(f)
    
print(data.shape)

from torchvision.utils import save_image

data = data.permute(1, 0, 2, 3)

for i, img in enumerate(data) :
    save_image(img, f'diving_0_frames_{i}_fixed.png')