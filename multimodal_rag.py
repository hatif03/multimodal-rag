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


dataset_folder = "./dataset/flowers"
os.makedirs(dataset_folder, exist_ok=True)


def save_images(dataset, dataset_folder, num_images=1000):
    for i in range(num_images):
        print(f"Saving image {i+1} of {num_images}")
        # get image
        image = dataset["train"][i]["image"]
        # save image
        image.save(os.path.join(dataset_folder, f"flower_{i+1}.png"))
    
    print(f"Saved the first 1000 images to : {dataset_folder}")


save_images(ds, dataset_folder, num_images=1000)