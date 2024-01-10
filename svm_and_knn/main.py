import cv2
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Download the dataset from the internet
url = "https://raw.githubusercontent.com/HoldenCaulfieldRye/caulfield_rye_share/master/emnist/emnist-letters-train-images-idx3-ubyte"
filename = "emnist-letters-train-images-idx3-ubyte"

# Save the dataset to the current directory
import urllib.request
urllib.request.urlretrieve(url, filename)

# Unzip the dataset
import tarfile
with tarfile.open(filename, 'r') as tar_ref:
    tar_ref.extractall()

# Load the images and labels from the dataset
emnist = datasets.emnist.load_data()
images, labels = emnist['data'], emnist['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Scale the data to be between -1 and 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM classifier
svm_classifier = SVC(kernel='linear', C=1)
svm_classifier.fit(X_train, y_train)

# Train the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

# Predict the labels for the test set
svm_predicted_labels = svm_classifier.predict(X_test)
knn_predicted_labels = knn_classifier.predict(X_test)

# Print the classification reports for the SVM and KNN classifiers
print("SVM Classifier Report:")
print(classification_report(y_test, svm_predicted_labels))

print("KNN Classifier Report:")
print(classification_report(y_test, knn_predicted_labels))

# Choose a random index from the test set
index = np.random.randint(0, len(X_test))

# Load the image using OpenCV
image = X_test[index].reshape(28, 28)
image = cv2.merge([image] * 3)

# Get the predicted label
predicted_label = knn_predicted_labels[index]

# Display the image and the predicted label
cv2.imshow('Sample Image', image)
print('Predicted Label:', predicted_label)
cv2.waitKey(0)
cv2.destroyAllWindows()
