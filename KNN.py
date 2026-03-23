import pandas as pd
from sklearn.svm import SVC


def pre_processing(df, target_col='Grade class 1: 90+  0:90-'):
    """
    Preprocess the wine dataset for Project 1 Phase 2.

    Assumptions:
    - First column is a wine-name identifier (not a predictor)
    - Target column is binary (0 or 1)
    - Remaining columns are already binary predictor features
    """

    df = df.copy()

    # Drop the identifier column (wine name column)
    df = df.iloc[:, 1:]

    # Separate target column
    y = df[target_col].astype(int)

    # Keep predictor columns only
    X = df.drop(columns=[target_col]).astype(int)

    return X, y


def dataframe_to_lists(X, y):
    """
    Convert pandas DataFrame and Series to Python lists
    for use with the custom KNN implementation.
    """

    x_values = X.to_numpy(dtype=int).tolist()
    y_values = y.tolist()

    return x_values, y_values


def jaccard_distance(train_data, test_data):
    """
    Compute Jaccard distance between two binary feature vectors.
    Used because all predictor features in this dataset are binary.
    """
    intersection = 0
    union = 0

    for i in range(len(train_data)):
        if train_data[i] == 1 and test_data[i] == 1:
            intersection += 1
        if train_data[i] == 1 or test_data[i] == 1:
            union += 1

    if union == 0:
        return 0

    return 1 - (intersection / union)


def k_nearest_neighbors(X_train, y_train, test_point, k):
    distances = []

    for i in range(len(X_train)):
        distance = jaccard_distance(X_train[i], test_point)
        distances.append((distance, y_train[i]))

    # Sort training points by distance from the test point (smallest distance first)
    distances.sort(key=lambda pair: pair[0])

    neighbors = distances[:k]
    return neighbors


def predict_class(neighbors):
    """
    Majority vote.
    If there is a tie, choose the class of the closest neighbor among the tied classes.
    """
    class_counts = {}

    for distance, label in neighbors:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    best_count = max(class_counts.values())
    tied_labels = [label for label, count in class_counts.items() if count == best_count]

    # No tie
    if len(tied_labels) == 1:
        return tied_labels[0]

    # Tie-break rule: choose the label of the closest neighbor among tied labels
    for distance, label in neighbors:
        if label in tied_labels:
            return label


def predict_all(X_train, y_train, X_test, k):
    predictions = []

    for test_point in X_test:
        neighbors = k_nearest_neighbors(X_train, y_train, test_point, k)
        prediction = predict_class(neighbors)
        predictions.append(prediction)

    return predictions


def confusion_matrix_values(y_true, y_pred):
    TP = FP = TN = FN = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1

    return TP, FP, FN, TN


def classification_metrics(y_true, y_pred):
    TP, FP, FN, TN = confusion_matrix_values(y_true, y_pred)

    total = TP + FP + FN + TN
    accuracy = (TP + TN) / total if total != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    }


def print_results(title, metrics):
    print(title)
    print("Accuracy:", metrics["Accuracy"])
    print("Precision:", metrics["Precision"])
    print("Sensitivity:", metrics["Sensitivity"])
    print("Specificity:", metrics["Specificity"])
    print()

    print("Confusion Matrix Values")
    print("TP:", metrics["TP"])
    print("FP:", metrics["FP"])
    print("FN:", metrics["FN"])
    print("TN:", metrics["TN"])
    print()


def print_label_predictions(y_true, y_pred):
    print("Real Label  Predicted Label")
    for real, pred in zip(y_true, y_pred):
        print(real, "        ", pred)
    print()


def svm_train_test(X_train, y_train, X_test, y_test):
    """
    Framework-based SVM for Phase 2 Part 4.
    """
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = classification_metrics(list(y_test), list(predictions))
    return list(y_test), list(predictions), metrics


if __name__ == "__main__":
    train_df = pd.read_csv("Training-dataset.csv", keep_default_na=False)
    test_df = pd.read_csv("Testing-dataset.csv", keep_default_na=False)

    target_col = 'Grade class 1: 90+  0:90-'

    # Use the provided split directly
    X_train, y_train = pre_processing(train_df, target_col)
    X_test, y_test = pre_processing(test_df, target_col)

    # Make sure test columns match train columns exactly
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # -----------------------------
    # PART 2 + PART 3: KNN (Custom)
    # -----------------------------
    # k is chosen ahead of time to avoid using the test set to choose the model parameter
    k = 5

    X_train_list, y_train_list = dataframe_to_lists(X_train, y_train)
    X_test_list, y_test_list = dataframe_to_lists(X_test, y_test)

    knn_predictions = predict_all(X_train_list, y_train_list, X_test_list, k)
    knn_metrics = classification_metrics(y_test_list, knn_predictions)

    print_results(f"KNN External Train/Test Results (90/10) using k={k}", knn_metrics)
    print_label_predictions(y_test_list, knn_predictions)

    # --------------------------------
    # PART 4: SVM (90% Train / 10% Test)
    # --------------------------------
    y_true_svm_90_10, predictions_svm_90_10, metrics_svm_90_10 = svm_train_test(
        X_train,
        y_train,
        X_test,
        y_test
    )

    print_results("SVM External Train/Test Results (90/10)", metrics_svm_90_10)
    print_label_predictions(y_true_svm_90_10, predictions_svm_90_10)

    # --------------------------------
    # PART 4: SVM (10% Train / 90% Test)
    # --------------------------------
    # Reverse the provided datasets for the 10/90 experiment:
    # train on the provided testing set and test on the provided training set.
    X_train_10 = X_test.copy()
    y_train_10 = y_test.copy()
    X_test_90 = X_train.copy()
    y_test_90 = y_train.copy()

    X_train_10 = X_train_10.reindex(columns=X_test_90.columns, fill_value=0)

    y_true_svm_10_90, predictions_svm_10_90, metrics_svm_10_90 = svm_train_test(
        X_train_10,
        y_train_10,
        X_test_90,
        y_test_90
    )

    print_results("SVM External Train/Test Results (10/90)", metrics_svm_10_90)
    print_label_predictions(y_true_svm_10_90, predictions_svm_10_90)