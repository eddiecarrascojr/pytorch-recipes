import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Step 1: Download from Kaggle
api = KaggleApi()
api.authenticate()
api.dataset_download_files('tongpython/cat-and-dog', path='data', unzip=True)

# Step 2: Organize into train/val folders
import shutil
from sklearn.model_selection import train_test_split

source_dir = "kaggle_data/cat_and_dog"
train_dir = "kaggle_data/train"
val_dir = "kaggle_data/val"
os.makedirs(train_dir + "/cats", exist_ok=True)
os.makedirs(train_dir + "/dogs", exist_ok=True)
os.makedirs(val_dir + "/cats", exist_ok=True)
os.makedirs(val_dir + "/dogs", exist_ok=True)

all_files = os.listdir(source_dir)
cat_files = [f for f in all_files if 'cat' in f]
dog_files = [f for f in all_files if 'dog' in f]

train_cats, val_cats = train_test_split(cat_files, test_size=0.2, random_state=42)
train_dogs, val_dogs = train_test_split(dog_files, test_size=0.2, random_state=42)

def copy_files(files, src, dst):
    for f in files:
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))

copy_files(train_cats, source_dir, train_dir + "/cats")
copy_files(val_cats, source_dir, val_dir + "/cats")
copy_files(train_dogs, source_dir, train_dir + "/dogs")
copy_files(val_dogs, source_dir, val_dir + "/dogs")
