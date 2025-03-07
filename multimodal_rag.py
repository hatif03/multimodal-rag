from dotenv import load_dotenv

from matplotlib import pyplot as plt
import os
from PIL import Image

load_dotenv()

from datasets import load_dataset

ds = load_dataset("huggan/flowers-102-categories")

print(ds.num_rows)

def show_image_from_uri(uri):
    img = Image.open(uri)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


