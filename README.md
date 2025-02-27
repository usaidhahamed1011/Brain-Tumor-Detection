# Brain Tumor Detection Using CNN
This project is a deep learning-based approach to detecting brain tumors from MRI images using Convolutional Neural Networks (CNNs). The model is trained on a dataset of labeled MRI scans and classifies images into tumor (yes) and no tumor (no) categories.

📌 Features
Preprocessing: Converts images to grayscale, resizes them, and applies histogram equalization for better contrast.
Data Augmentation: Uses techniques like rotation, shifting, zooming, and flipping to improve model generalization.
CNN Model: A sequential deep learning model with Convolutional, Pooling, Dropout, and Dense layers.
Training & Validation: Splits the dataset into training and testing sets (80:20) for model evaluation.
Visualization: Displays random test images with model predictions and confidence scores.
Prediction Function: Allows classification of new MRI images.

📂 Dataset
The dataset consists of MRI scan images categorized into:

yes/ → Images containing brain tumors.
no/ → Images without brain tumors.

│── yes/  (Images with tumors)
│── no/   (Images without tumors)
Ensure the dataset is stored correctly before running the script.

🛠️ Installation & Setup
1️⃣ Install Required Libraries
Run the following command to install dependencies:
pip install tensorflow numpy opencv-python matplotlib scikit-learn

2️⃣ Clone the Repository
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection

3️⃣ Run the Project
python brain_tumor_detection.py

🖼️ Sample Predictions
MRI Image	Prediction	Confidence
Tumor	98.4%
No Tumor	95.6%

📌 Future Improvements
Train on a larger dataset for better generalization.
Experiment with different architectures like ResNet or MobileNet.
Deploy the model using Flask or FastAPI for real-time detection.
