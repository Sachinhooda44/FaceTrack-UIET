import os
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

print("Loading face embeddings...")
embeddings_dir = "embeddings"
known_embeddings = []
known_names = []

# Load all embeddings and their corresponding names
for name in os.listdir(embeddings_dir):
    person_dir = os.path.join(embeddings_dir, name)
    if not os.path.isdir(person_dir):
        continue
    for file in os.listdir(person_dir):
        emb_path = os.path.join(person_dir, file)
        emb = np.load(emb_path)
        known_embeddings.append(emb)
        known_names.append(name)

if not known_embeddings:
    print("No embeddings found to train on. Please run the data collection script first.")
else:
    # Convert face embeddings list to a NumPy array
    known_embeddings = np.array(known_embeddings)

    # Encode the text labels (names) into numbers
    print("Encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(known_names)

    # Train the SVM classifier
    print("Training the classifier...")
    # The 'probability=True' argument is crucial for getting confidence scores
    classifier = SVC(C=1.0, kernel='linear', probability=True)
    classifier.fit(known_embeddings, labels)

    # Save the trained classifier to a file
    with open('classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)

    # Save the label encoder to a file
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
        
    print("Training complete. Classifier and label encoder saved.")