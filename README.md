# 🩻 Automated Chest X-Ray Classification

This project uses a Convolutional Neural Network (CNN) to classify chest X-ray images into four categories: **Normal**, **Pneumonia**, **COVID-19**, and **Tuberculosis**.

---

##  Project Structure

- `download_dataset.py` – Script to download and organize datasets
- `preprocess.py` – Image preprocessing & augmentation
- `train_model.py` – Model architecture & training
- `evaluate.py` – Evaluation of test data
- `prediction.py` – Run predictions on new images
- `confusion.py` – Plot confusion matrix
- `classification_report.py` – Generate precision/recall/F1 report
- `cnn_architecture_vis.py` – Visualize model architecture
- `Loss Graphs.py` – Plot training & validation loss/accuracy
- `project.py` – End-to-end runner (combine all steps)

---

## How to Use

### 1.  Download the Dataset

Dataset is **not included** in this repo. Use `download_dataset.py` or manually download datasets from:

- [NIH Chest X-ray](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- [Tuberculosis Chest X-ray Dataset](https://www.kaggle.com/datasets/andrewmvd/tuberculosis-chest-xray-dataset)



---

### 2.  Preprocess the Data

python preprocess.py

python train_model.py

python evaluate.py


python classification_report.py     # Precision, Recall, F1
python confusion.py                 # Confusion matrix
python cnn_architecture_vis.py      # Model summary/plot
python "Loss Graphs.py"             # Accuracy/loss graphs


python prediction.py --image test_cases/sample.jpg


