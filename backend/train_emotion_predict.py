# train_emotion_predict.py

import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import extract
import categorize

# Lists for storing the full feature vector and the emotion labels
X = []         # Each element: [baseline_angle, top_margin, letter_size, line_spacing, word_spacing, pen_pressure, slant_angle]
y_emotion = [] # Emotion label (string, e.g., "Depressed", "Happy", etc.)
page_ids = []  # Optionally store the page id

if os.path.isfile("emotion_label_list"):
    print("Info: emotion_label_list found.")

    # Read all lines from the emotion label file
    with open("emotion_label_list", "r") as f:
        lines = f.readlines()

    total_samples = len(lines)
    print(f"Total samples to process: {total_samples}")

    # Each line is expected to have:
    # baseline_angle, top_margin, letter_size, line_spacing, word_spacing, pen_pressure, slant_angle, emotion_label, page_id
    for line in lines:
        content = line.split()
        # Parse the first seven features as floats
        features = [float(x) for x in content[0:7]]
        emotion = content[7]  # emotion label
        page_id = content[8] if len(content) > 8 else ""
        X.append(features)
        y_emotion.append(emotion)
        page_ids.append(page_id)

    # Print out the first few samples for inspection
    print("Sample features and emotion labels:")
    for i in range(min(5, total_samples)):
        print(f"Sample {i+1}: Features: {X[i]}, Emotion: {y_emotion[i]}")

    # --- Optimization Step: Feature Scaling and Hyperparameter Tuning ---
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_emotion, test_size=0.30, random_state=42)

    # Define a parameter grid for SVC optimization using GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5)
    grid.fit(X_train, y_train)

    print("Best parameters found:", grid.best_params_)
    clf = grid.best_estimator_

    test_accuracy = accuracy_score(clf.predict(X_test), y_test)
    print("Emotion Classifier Accuracy:", test_accuracy)

    # --- Interactive Prediction ---
    while True:
        file_name = input("Enter file name to predict emotion (or 'z' to exit): ")
        if file_name.lower() == 'z':
            break

        # Extract raw features from the new sample
        raw_features = extract.start(file_name)

        # Categorize each raw feature using the categorize module functions
        baseline_angle, _ = categorize.determine_baseline_angle(raw_features[0])
        top_margin, _ = categorize.determine_top_margin(raw_features[1])
        letter_size, _ = categorize.determine_letter_size(raw_features[2])
        line_spacing, _ = categorize.determine_line_spacing(raw_features[3])
        word_spacing, _ = categorize.determine_word_spacing(raw_features[4])
        pen_pressure, _ = categorize.determine_pen_pressure(raw_features[5])
        slant_angle, _ = categorize.determine_slant_angle(raw_features[6])

        # Build the feature vector (order must match training)
        features_vector = [baseline_angle, top_margin, letter_size, line_spacing, word_spacing, pen_pressure, slant_angle]
        print("Extracted features:", features_vector)

        # Scale the feature vector using the previously fitted scaler
        features_vector_scaled = scaler.transform([features_vector])
        predicted_emotion = clf.predict(features_vector_scaled)
        print("Predicted Emotion:", predicted_emotion[0])
        print("---------------------------------------------------")

else:
    print("Error: emotion_label_list file not found.")
