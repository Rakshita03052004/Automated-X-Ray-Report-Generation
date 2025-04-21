
import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_data():
    api = KaggleApi()
    api.authenticate()

    if not os.path.exists("chest_xray"):
        os.makedirs("chest_xray")

    print("Downloading dataset...")
    api.dataset_download_files('prashant268/chest-xray-covid19-pneumonia-tuberculosis', path='chest_xray', unzip=True)
    print("Download complete!")

if __name__ == "__main__":
    download_data()
