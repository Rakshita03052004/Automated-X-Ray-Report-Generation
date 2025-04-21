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
o	Loss Function: Specify the loss function used for training (e.g., Categorical Crossentropy, especially suitable for multi-class classification with one-hot encoded labels). Explain why this loss function was chosen.
o	Optimizer: Detail the optimizer used (e.g., Adam, SGD, RMSprop). Specify the learning rate used and the reasons for choosing this initial value. Mention any learning rate scheduling techniques applied (e.g., reducing the learning rate on plateau).
o	Batch Size: State the batch size used during training and explain the rationale behind this choice (e.g., balancing between training speed and gradient stability).
o	Number of Epochs: Specify the total number of training epochs.
o	Early Stopping: Explain if early stopping was implemented to prevent overfitting. Describe the criteria used for early stopping (e.g., monitoring validation loss and stopping if it doesn't improve for a certain number of epochs).
o	Regularization Techniques: Describe any other regularization techniques used (e.g., L1 or L2 regularization applied to the weights).
•	Evaluation Metrics: Reiterate the evaluation metrics used (Accuracy, Precision, Recall, F1-score, Confusion Matrix) and explain how each metric provides insight into the model's performance, particularly in the context of medical diagnosis (e.g., the importance of high recall to avoid missing positive cases).






