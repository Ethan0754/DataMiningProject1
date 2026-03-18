import numpy as np
import pandas as pd
import random
from svm_model import svm_80_20, svm_k_fold, print_svm_results
from itertools import combinations


def pre_processing(df, target_col='Grade class 1: 90+  0:90-', feature_columns=None):
    df = df.copy()

    # Drop identifier-like columns
    drop_cols = ['Unnamed: 0']
    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    df.drop(columns=existing_drop_cols, inplace=True)

    if feature_columns is not None:
        keep_cols = [col for col in feature_columns if col in df.columns]
        if target_col in df.columns:
            df = df[[target_col] + keep_cols].copy()
        else:
            df = df[keep_cols].copy()

    # Separate target column
    y = None
    if target_col in df.columns:
        y = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
        df.drop(columns=[target_col], inplace=True)

    # Ensure all features are numeric
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

    # shuffle the dataset randomly
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # split the data by training_percent
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

    # find the distance from each point in x_train to test_point
    for i in range(len(X_train)):
        distance = jaccard_distance(X_train[i], test_point)
        distances.append((distance, y_train[i]))

    distances.sort(key=lambda pair: pair[0])  # sort by distance

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

    for test_point in X_test:  # using test data, return the predictions for each line in x_test
        neighbor_labels = k_nearest_neighbors(X_train, y_train, test_point, k)
        prediction = predict_class(neighbor_labels)
        predictions.append(prediction)

    return predictions


def make_folds_df(X, y, k):
    data = X.copy()
    data['target'] = y

    # shuffle the dataset randomly
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    folds = []
    fold_size = len(data) // k  # floor division

    # isolate the folds into their own index of a list (folds)
    start = 0
    for i in range(k):
        if i == k - 1:
            fold = data.iloc[start:].reset_index(drop=True)
        else:
            fold = data.iloc[start:start + fold_size].reset_index(drop=True)

        folds.append(fold)
        start += fold_size

    return folds


def get_train_test_from_folds_df(folds, test_fold_index):
    """Function is the same as train_test_split, but for folds instead of 80/20 split"""
    test_data = folds[test_fold_index]
    train_parts = []

    for i in range(len(folds)):
        if i != test_fold_index:
            train_parts.append(folds[i])

    train_data = pd.concat(train_parts, ignore_index=True)

    X_train = train_data.drop(columns=['target']).reset_index(drop=True)
    y_train = train_data['target'].reset_index(drop=True)

    X_test = test_data.drop(columns=['target']).reset_index(drop=True)
    y_test = test_data['target'].reset_index(drop=True)

    return X_train, y_train, X_test, y_test


def accuracy_score(y_true, y_pred):
    correct = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:  # Count the amount of correct predictions
            correct += 1

    return correct / len(y_true)  # Find correctness percentage


def run_k_fold_df(X, y, num_folds, k):
    """Same as predict_all but for folds"""
    folds = make_folds_df(X, y, num_folds)
    scores = []

    for i in range(num_folds):
        X_train, y_train, X_test, y_test = get_train_test_from_folds_df(folds, i)

        # convert dataframes to lists so KNN functions can use them
        X_train_list, y_train_list = dataframe_to_lists(X_train, y_train)
        X_test_list, y_test_list = dataframe_to_lists(X_test, y_test)

        predictions = predict_all(X_train_list, y_train_list, X_test_list, k)
        accuracy = accuracy_score(y_test_list, predictions)
        scores.append(accuracy)

    average_accuracy = sum(scores) / len(scores)
    return scores, average_accuracy


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

    accuracy = (TP + TN) / (TP + TN + FP + FN)
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


if __name__ == "__main__":
    train_df = pd.read_csv(
        "Training-dataset.csv",
        keep_default_na=False
    )

    test_df = pd.read_csv(
        "Testing-dataset.csv",
        keep_default_na=False
    )

    target_col = 'Grade class 1: 90+  0:90-'

    X, y = pre_processing(train_df, target_col)
    X_external_test, y_external_test = pre_processing(test_df, target_col, feature_columns=X.columns.tolist())
    X_external_test = X_external_test.reindex(columns=X.columns, fill_value=0)

    num_folds1 = 5
    num_folds2 = 10

    # External train/test
    X_train = X.copy()
    y_train = y.copy()
    X_test = X_external_test.copy()
    y_test = y_external_test.copy()

    # convert dataframes to lists so KNN functions can use them
    X_train_list, y_train_list = dataframe_to_lists(X_train, y_train)
    X_test_list, y_test_list = dataframe_to_lists(X_test, y_test)

    # test multiple k values, but only keep the best result
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

    print("KNN External Train/Test Results")
    print("Best k:", best_k)
    print("Accuracy:", best_metrics["Accuracy"])
    print("Precision:", best_metrics["Precision"])
    print("Sensitivity:", best_metrics["Sensitivity"])
    print("Specificity:", best_metrics["Specificity"])

    print("\nConfusion Matrix Values")
    print("TP:", best_metrics["TP"])
    print("FP:", best_metrics["FP"])
    print("FN:", best_metrics["FN"])
    print("TN:", best_metrics["TN"])
    print()

    print("Real Label  Predicted Label")
    for real, pred in zip(y_test_list, best_predictions):
        print(real, "        ", pred)
    print()

    # 5-fold cross validation using best k
    folds_5 = make_folds_df(X, y, num_folds1)
    scores_5 = []
    all_y_true_5 = []
    all_y_pred_5 = []

    for i in range(num_folds1):
        X_train, y_train, X_test, y_test = get_train_test_from_folds_df(folds_5, i)

        X_train_list, y_train_list = dataframe_to_lists(X_train, y_train)
        X_test_list, y_test_list = dataframe_to_lists(X_test, y_test)

        predictions = predict_all(X_train_list, y_train_list, X_test_list, best_k)
        accuracy = accuracy_score(y_test_list, predictions)

        scores_5.append(accuracy)
        all_y_true_5.extend(y_test_list)
        all_y_pred_5.extend(predictions)

    average_5 = sum(scores_5) / len(scores_5)
    metrics_5 = classification_metrics(all_y_true_5, all_y_pred_5)

    print("KNN 5-Fold Cross Validation")
    print("Best k used:", best_k)
    print("Fold Accuracies:", scores_5)
    print("Average Accuracy:", average_5)
    print("Precision:", metrics_5["Precision"])
    print("Sensitivity:", metrics_5["Sensitivity"])
    print("Specificity:", metrics_5["Specificity"])

    print("\nConfusion Matrix Values")
    print("TP:", metrics_5["TP"])
    print("FP:", metrics_5["FP"])
    print("FN:", metrics_5["FN"])
    print("TN:", metrics_5["TN"])
    print()

    # 10-fold cross validation using best k
    folds_10 = make_folds_df(X, y, num_folds2)
    scores_10 = []
    all_y_true_10 = []
    all_y_pred_10 = []

    for i in range(num_folds2):
        X_train, y_train, X_test, y_test = get_train_test_from_folds_df(folds_10, i)

        X_train_list, y_train_list = dataframe_to_lists(X_train, y_train)
        X_test_list, y_test_list = dataframe_to_lists(X_test, y_test)

        predictions = predict_all(X_train_list, y_train_list, X_test_list, best_k)
        accuracy = accuracy_score(y_test_list, predictions)

        scores_10.append(accuracy)
        all_y_true_10.extend(y_test_list)
        all_y_pred_10.extend(predictions)

    average_10 = sum(scores_10) / len(scores_10)
    metrics_10 = classification_metrics(all_y_true_10, all_y_pred_10)

    print("KNN 10-Fold Cross Validation")
    print("Best k used:", best_k)
    print("Fold Accuracies:", scores_10)
    print("Average Accuracy:", average_10)
    print("Precision:", metrics_10["Precision"])
    print("Sensitivity:", metrics_10["Sensitivity"])
    print("Specificity:", metrics_10["Specificity"])

    print("\nConfusion Matrix Values")
    print("TP:", metrics_10["TP"])
    print("FP:", metrics_10["FP"])
    print("FN:", metrics_10["FN"])
    print("TN:", metrics_10["TN"])
    print()

    # KNN test using only the two most important features
    important_features = X.var().sort_values(ascending=False).head(2).index.tolist()
    X_two_features = X[important_features].copy()
    X_test_two_features = X_external_test[important_features].copy()

    X_train = X_two_features.copy()
    y_train = y.copy()
    X_test = X_test_two_features.copy()
    y_test = y_external_test.copy()

    X_train_list, y_train_list = dataframe_to_lists(X_train, y_train)
    X_test_list, y_test_list = dataframe_to_lists(X_test, y_test)

    best_k_two_features = None
    best_accuracy_two_features = -1
    best_predictions_two_features = None
    best_metrics_two_features = None

    for k in k_values:
        predictions = predict_all(X_train_list, y_train_list, X_test_list, k)
        metrics = classification_metrics(y_test_list, predictions)
        accuracy = metrics["Accuracy"]

        if accuracy > best_accuracy_two_features:
            best_accuracy_two_features = accuracy
            best_k_two_features = k
            best_predictions_two_features = predictions
            best_metrics_two_features = metrics

    print(f"KNN External Train/Test Using Only {important_features[0]} and {important_features[1]}")
    print("Best k:", best_k_two_features)
    print("Accuracy:", best_metrics_two_features["Accuracy"])
    print("Precision:", best_metrics_two_features["Precision"])
    print("Sensitivity:", best_metrics_two_features["Sensitivity"])
    print("Specificity:", best_metrics_two_features["Specificity"])

    print("\nConfusion Matrix Values")
    print("TP:", best_metrics_two_features["TP"])
    print("FP:", best_metrics_two_features["FP"])
    print("FN:", best_metrics_two_features["FN"])
    print("TN:", best_metrics_two_features["TN"])
    print()

    print("Real Label  Predicted Label")
    for real, pred in zip(y_test_list, best_predictions_two_features):
        print(real, "        ", pred)
    print()

    # SVM 80/20 (Training/Test)
    y_test, predictions, metrics = svm_80_20(
        X,
        y,
        X_external_test,
        y_external_test,
        train_test_split_df,
        no_normalization,
        classification_metrics
    )

    print_svm_results("SVM External Train/Test Results", metrics)

    print("Real Label  Predicted Label")
    for real, pred in zip(y_test, predictions):
        print(real, "       ", pred)

    print()

    # SVM 20/80 (Training/Test)
   # y_test, predictions, metrics = svm_20_80(
       # X,
       # y,
       # X_external_test,
       # y_external_test,
       # train_test_split_df,
       # no_normalization,
       # classification_metrics
    #)

    #print_svm_results("SVM External Train/Test Results", metrics)

    # SVM 5-fold Cross Validation
    scores_5, metrics_5 = svm_k_fold(
        X,
        y,
        num_folds1,
        make_folds_df,
        get_train_test_from_folds_df,
        no_normalization,
        classification_metrics
    )

    print("SVM 5-Fold Cross Validation")
    print("Fold Accuracies:", scores_5)
    print_svm_results("Average 5-Fold Metrics", metrics_5)

    # SVM 10-fold Cross Validation
    scores_10, metrics_10 = svm_k_fold(
        X,
        y,
        num_folds2,
        make_folds_df,
        get_train_test_from_folds_df,
        no_normalization,
        classification_metrics
    )

    print("SVM 10-Fold Cross Validation")
    print("Fold Accuracies:", scores_10)
    print_svm_results("Average 10-Fold Metrics", metrics_10)