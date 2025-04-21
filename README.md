# 🩻 Automated Chest X-Ray Classification

This project uses a Convolutional Neural Network (CNN) to classify chest X-ray images into four categories: **Normal**, **Pneumonia**, **COVID-19**, and **Tuberculosis**.

---

##  Project Structure

- `download_dataset.py` – Script to download and organize datasets and extract features 
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
Learning
A deep learning-based system to detect respiratory diseases — Pneumonia, COVID-19, and Tuberculosis — from chest X-ray images using a Convolutional Neural Network (CNN). This project aims to assist radiologists and medical professionals, particularly in remote or understaffed regions, by providing an AI-powered preliminary diagnosis tool.

Features
Classifies X-rays into: Normal, Pneumonia, COVID-19, Tuberculosis
Custom CNN-based image classifier
Grad-CAM for visual interpretability
Evaluation metrics: Accuracy, Precision, Recall, F1-Score
Supports doctors in rapid, automated screening

Tech Stack
Language: Python 3.x
Frameworks: TensorFlow, Keras
Libraries: NumPy, Pandas, OpenCV, Matplotlib, Seaborn, scikit-learn
Development: Jupyter Notebook / Google Colab


#Installation
Copy code
Clone the repo
git clone https://github.com/your-username/xray-disease-classifier.git
cd xray-disease-classifier
(Optional) Create a virtual environment  python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the notebook or script
jupyter notebook  # or open in Google Colab

Preprocessed with:
Image resizing to 224x224
Normalization to [0, 1]
Data Augmentation: rotation, flipping, zoom, contrast

•	Model Architecture: 
o	Custom CNN: If you designed a CNN from scratch, provide a detailed description of each layer in the network: 
	Specify the type of layer (Convolutional 2D, MaxPooling 2D, Flatten, Dense, Dropout).
	For Convolutional layers: Number of filters, kernel size, stride, padding, activation function (e.g., ReLU).
	For MaxPooling layers: Pool size, stride.
	For Dense layers: Number of neurons, activation function.
	For Dropout layers: Dropout rate.
	For the Output layer: Number of neurons (equal to the number of classes - 4), activation function (Softmax for multi-class probability distribution).
	Provide a visual representation of the network architecture (e.g., a diagram).



•	Training Details: 
Optimizer: Adam  Loss: Categorical Crossentropy   Epochs: 50  batch Size: 32

Early Stopping to prevent overfitting
o	Loss Function: Specify the loss function used for training (e.g., Categorical Crossentropy, especially suitable for multi-class classification with one-hot encoded labels). Explain why this loss function was chosen.
o	Optimizer: Detail the optimizer used (e.g., Adam, SGD, RMSprop). Specify the learning rate used and the reasons for choosing this initial value. Mention any learning rate scheduling techniques applied (e.g., reducing the learning rate on plateau).
o	Batch Size: State the batch size used during training and explain the rationale behind this choice (e.g., balancing between training speed and gradient stability).
o	Number of Epochs: Specify the total number of training epochs.
o	Early Stopping: Explain if early stopping was implemented to prevent overfitting. Describe the criteria used for early stopping (e.g., monitoring validation loss and stopping if it doesn't improve for a certain number of epochs).
o	Regularization Techniques: Describe any other regularization techniques used (e.g., L1 or L2 regularization applied to the weights).
•	Evaluation Metrics: Reiterate the evaluation metrics used (Accuracy, Precision, Recall, F1-score, Confusion Matrix) and explain how each metric provides insight into the model's performance, particularly in the context of medical diagnosis (e.g., the importance of high recall to avoid missing positive cases).


•	Sample X-ray images (Normal & Disease)
![image](https://github.com/user-attachments/assets/17c9504e-f959-4b44-a2ab-bfe52918976f)


•	CNN architecture visualization
 ![image](https://github.com/user-attachments/assets/17d5a635-113f-4649-83aa-efea08c2159a)

 
•	Training accuracy/loss graphs 
 ![image](https://github.com/user-attachments/assets/147521d9-730c-4d10-90cd-b7e7617b63b1)
![image](https://github.com/user-attachments/assets/eda460a1-f7ad-4b8e-adbe-50eadffbd8cc)


•	Confusion matrix
![image](https://github.com/user-attachments/assets/543a2113-132a-42cc-b697-c4bb31466db7)

normal image and prediction 
•	Predicted results for test images
![image](https://github.com/user-attachments/assets/9e3c8999-7545-4e9a-9878-be29b5b557aa)
![image](https://github.com/user-attachments/assets/44e72bd3-0bb3-47a3-8daf-61b1cfa2e7b9)

pneumonia image and prediction 
 ![image](https://github.com/user-attachments/assets/c2ca6602-0b00-4dcd-b8ff-9c837a2553a1)
![image](https://github.com/user-attachments/assets/7cebade0-ab07-4947-96d1-0c02b6b41c49)




 
 










