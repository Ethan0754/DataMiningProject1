import pandas as pd
from sklearn.svm import SVC


def pre_processing(df, target_col='Grade class 1: 90+  0:90-', feature_columns=None):
    df = df.copy()

    # Drop common index/id columns if present
    drop_cols = ['Unnamed: 0']
    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    df.drop(columns=existing_drop_cols, inplace=True)

    # Separate target first
    y = None
    if target_col in df.columns:
        y = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
        df.drop(columns=[target_col], inplace=True)

    # If specific feature columns are provided, keep only those
    if feature_columns is not None:
        keep_cols = [col for col in feature_columns if col in df.columns]
        df = df[keep_cols].copy()

    # Drop non-numeric / identifier-like columns
    # This removes the first wine-name column shown in your screenshot.
    non_numeric_cols = [
        col for col in df.columns
        if not pd.api.types.is_numeric_dtype(df[col])
    ]
    df.drop(columns=non_numeric_cols, inplace=True)

    # Convert all remaining features to numeric just in case
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values
    df = df.fillna(0)

    return df.astype(float), y


def no_normalization(X_train, X_test):
    return X_train.copy(), X_test.copy()


def train_test_split_df(X, y, train_percent=0.8):
    data = X.copy()
    data['target'] = y

    # Shuffle dataset
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    split_index = int(len(data) * train_percent)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    X_train = train_data.drop(columns=['target']).reset_index(drop=True)
    y_train = train_data['target'].reset_index(drop=True)

    X_test = test_data.drop(columns=['target']).reset_index(drop=True)
    y_test = test_data['target'].reset_index(drop=True)

    return X_train, y_train, X_test, y_test


def dataframe_to_lists(X, y):
    X_values = X.to_numpy(dtype=float).tolist()
    y_values = y.tolist()
    return X_values, y_values


def jaccard_distance(train_data, test_data):
    intersection = 0
    union = 0

    for i in range(len(train_data)):
        a = 1 if train_data[i] != 0 else 0
        b = 1 if test_data[i] != 0 else 0

        if a == 1 and b == 1:
            intersection += 1
        if a == 1 or b == 1:
            union += 1

    if union == 0:
        return 0

    return 1 - (intersection / union)


def k_nearest_neighbors(X_train, y_train, test_point, k):
    distances = []

    for i in range(len(X_train)):
        distance = jaccard_distance(X_train[i], test_point)
        distances.append((distance, y_train[i]))

    distances.sort(key=lambda pair: pair[0])
    neighbors = distances[:k]
    return neighbors


def predict_class(neighbors):
    class_counts = {}

    for distance, label in neighbors:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    best_label = None
    best_count = -1

    for label in class_counts:
        if class_counts[label] > best_count:
            best_count = class_counts[label]
            best_label = label

    return best_label


def predict_all(X_train, y_train, X_test, k):
    predictions = []

    for test_point in X_test:
        neighbor_labels = k_nearest_neighbors(X_train, y_train, test_point, k)
        prediction = predict_class(neighbor_labels)
        predictions.append(prediction)

    return predictions


def accuracy_score(y_true, y_pred):
    correct = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1

    return correct / len(y_true)


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

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
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

    # Preprocess training data
    X_train_full, y_train_full = pre_processing(train_df, target_col)

    # Preprocess testing data using same feature columns
    X_test_external, y_test_external = pre_processing(
        test_df,
        target_col,
        feature_columns=X_train_full.columns.tolist()
    )

    # Make sure testing columns match training columns exactly
    X_test_external = X_test_external.reindex(columns=X_train_full.columns, fill_value=0)

    # -----------------------------
    # PART 2 + PART 3: KNN (Custom)
    # -----------------------------
    X_train_list, y_train_list = dataframe_to_lists(X_train_full, y_train_full)
    X_test_list, y_test_list = dataframe_to_lists(X_test_external, y_test_external)

    k_values = [1, 3, 5, 7, 9, 11, 15, 21]

    best_k = None
    best_accuracy = -1
    best_predictions = None
    best_metrics = None

    for k in k_values:
        predictions = predict_all(X_train_list, y_train_list, X_test_list, k)
        metrics = classification_metrics(y_test_list, predictions)
        accuracy = metrics["Accuracy"]

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_predictions = predictions
            best_metrics = metrics

    print("KNN External Train/Test Results (90/10)")
    print("Best k:", best_k)
    print("Accuracy:", best_metrics["Accuracy"])
    print("Precision:", best_metrics["Precision"])
    print("Sensitivity:", best_metrics["Sensitivity"])
    print("Specificity:", best_metrics["Specificity"])
    print()

    print("Confusion Matrix Values")
    print("TP:", best_metrics["TP"])
    print("FP:", best_metrics["FP"])
    print("FN:", best_metrics["FN"])
    print("TN:", best_metrics["TN"])
    print()

    print("Real Label  Predicted Label")
    for real, pred in zip(y_test_list, best_predictions):
        print(real, "        ", pred)
    print()

    # --------------------------------
    # PART 4: SVM (90% Train / 10% Test)
    # --------------------------------
    y_true_svm_90_10, predictions_svm_90_10, metrics_svm_90_10 = svm_train_test(
        X_train_full,
        y_train_full,
        X_test_external,
        y_test_external
    )

    print_results("SVM External Train/Test Results (90/10)", metrics_svm_90_10)

    print("Real Label  Predicted Label")
    for real, pred in zip(y_true_svm_90_10, predictions_svm_90_10):
        print(real, "        ", pred)
    print()

    # --------------------------------
    # PART 4: SVM (10% Train / 90% Test)
    # --------------------------------
    # Combine training + testing datasets so we can create a new 10/90 split
    X_all = pd.concat([X_train_full, X_test_external], ignore_index=True)
    y_all = pd.concat([y_train_full, y_test_external], ignore_index=True)

    X_train_10, y_train_10, X_test_90, y_test_90 = train_test_split_df(X_all, y_all, train_percent=0.10)

    y_true_svm_10_90, predictions_svm_10_90, metrics_svm_10_90 = svm_train_test(
        X_train_10,
        y_train_10,
        X_test_90,
        y_test_90
    )

    print_results("SVM Train/Test Results (10/90)", metrics_svm_10_90)

    print("Real Label  Predicted Label")
    for real, pred in zip(y_true_svm_10_90, predictions_svm_10_90):
        print(real, "        ", pred)
    print()