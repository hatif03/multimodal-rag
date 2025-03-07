from dotenv import load_dotenv

from datasets import load_dataset

ds = load_dataset("huggan/flowers-102-categories")

print(ds.num_rows)