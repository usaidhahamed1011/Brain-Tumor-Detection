import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = r"C:\Users\usius\OneDrive\Desktop\brainn\brain_tumor_dataset"
categories = ["yes", "no"]

def load_data(data_dir, categories, img_size=128):
    data = []
    labels = []
    for category in categories:
        folder_path = os.path.join(data_dir, category)
        label = categories.index(category)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (img_size, img_size))
                img = cv2.equalizeHist(img)
                data.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image: {e}")
    return np.array(data), np.array(labels)

IMG_SIZE = 128
X, y = load_data(data_dir, categories, IMG_SIZE)

X = X / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.summary()

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

history = model.fit(datagen.flow(X_train, y_train, batch_size=32),validation_data=(X_test, y_test),epochs=20)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

def predict_image(input_data):
    if isinstance(input_data, str): 
        img = cv2.imread(input_data)
        if img is None:
            raise FileNotFoundError(f"Image not found at {input_data}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = input_data
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(img)
    return categories[np.argmax(prediction)], np.max(prediction)

example_image_path = r"C:\Users\usius\OneDrive\Desktop\brainn\brain_tumor_dataset\yes\Y7.jpg" 
label, confidence = predict_image(example_image_path)
print(f"Prediction: {label}, Confidence: {confidence * 100:.2f}%")

def visualize_predictions(images, labels):
    plt.figure(figsize=(12, 12))
    for i in range(9):
        idx = np.random.randint(0, len(images))
        img = images[idx].reshape(IMG_SIZE, IMG_SIZE)
        pred_label, confidence = predict_image(img)  
        true_label = categories[np.argmax(labels[idx])]
        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence * 100:.1f}%)")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_predictions(X_test, y_test)
