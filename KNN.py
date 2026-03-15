import numpy as np
import pandas as pd
import random
from svm_model import svm_80_20, svm_20_80, svm_k_fold, print_svm_results
from itertools import combinations


def pre_processing(df, target_col='Purchase_Intent'):
    df = df.copy()

    # Drop identifier-like columns
    drop_cols = ['Customer_ID', 'Location']
    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    df.drop(columns=existing_drop_cols, inplace=True)

    # Clean Purchase_Amount column
    if 'Purchase_Amount' in df.columns:
        df['Purchase_Amount'] = (
            df['Purchase_Amount']
            .replace(r'[\$, ]', '', regex=True)
            .astype(float)
        )

    # Convert Time_of_Purchase -> day of week
    if 'Time_of_Purchase' in df.columns:
        df['Time_of_Purchase'] = pd.to_datetime(df['Time_of_Purchase'], errors='coerce')
        df['Time_of_Purchase'] = df['Time_of_Purchase'].dt.dayofweek

    # Convert boolean columns to binary
    bool_cols = ['Discount_Used', 'Customer_Loyalty_Program_Member']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Convert numeric columns
    numeric_cols = [
        'Age',
        'Purchase_Amount',
        'Frequency_of_Purchase',
        'Brand_Loyalty',
        'Product_Rating',
        'Time_Spent_on_Product_Research(hours)',
        'Return_Rate',
        'Customer_Satisfaction',
        'Time_to_Decision',
        'Time_of_Purchase'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Separate target column
    y = None
    if target_col in df.columns:
        y = df[target_col].copy()
        df.drop(columns=[target_col], inplace=True)

    # Automatically encode all categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        unique_vals = df[col].dropna().unique()
        mapping = {val: i for i, val in enumerate(unique_vals)}
        df[col] = df[col].map(mapping)


    # Fill missing values
    df = df.fillna(0)

    # Encode target
    if y is not None:
        if target_col == 'Purchase_Intent':
            y = y.map({
                'Need-based': 0,
                'Planned': 0,
                'Wants-based': 1,
                'Impulsive': 1
            })
        else:
            y = pd.Categorical(y).codes

    return df.astype(float), y

def threshold_normalize_df(X):
    X = X.copy()

    for col in X.columns:
        X[col] = (X[col] >= 0.5).astype(int)

    return X

def train_test_split_df(X, y, train_percent=0.8):
    data = X.copy()
    data['target'] = y

    # shuffle the dataset randomly
    data = data.sample(frac=1).reset_index(drop=True)

    # split the data by training_percent
    split_index = int(len(data) * train_percent)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    X_train = train_data.drop(columns=['target']).reset_index(drop=True)
    y_train = train_data['target'].reset_index(drop=True)

    X_test = test_data.drop(columns=['target']).reset_index(drop=True)
    y_test = test_data['target'].reset_index(drop=True)

    return X_train, y_train, X_test, y_test


def normalize_train_test_df(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in X_train.columns:

        min_val = X_train[col].min()
        max_val = X_train[col].max()

        # Avoid division by zero
        if max_val == min_val:
            X_train[col] = 0
            X_test[col] = 0
        else:
            X_train[col] = (X_train[col] - min_val) / (max_val - min_val)
            X_test[col] = (X_test[col] - min_val) / (max_val - min_val)

    return X_train, X_test


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
    class_scores = {}

    for distance, label in neighbors:

        weight = 1 / (distance + 1e-9)

        if label not in class_scores:
            class_scores[label] = 0

        class_scores[label] += weight

    best_label = None
    best_score = -1

    for label in class_scores:
        if class_scores[label] > best_score:
            best_score = class_scores[label]
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
    data = data.sample(frac=1).reset_index(drop=True)

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

        # normalize training and test data for this fold
        X_train, X_test = normalize_train_test_df(X_train, X_test)

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
    df = pd.read_csv(
        "Group 5-Ecommerce_Consumer_Behavior_Analysis_Data.csv",
        keep_default_na=False
    )

    X, y = pre_processing(df, 'Purchase_Intent')

    num_folds1 = 5
    num_folds2 = 10

    # 80/20 split
    X_train, y_train, X_test, y_test = train_test_split_df(X, y, train_percent=0.8)

    # normalize training and test data
    X_train, X_test = normalize_train_test_df(X_train, X_test)
    X_train = threshold_normalize_df(X_train)
    X_test = threshold_normalize_df(X_test)

    X_train.to_csv("PostProcessing.csv", index=False)

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

    print("KNN Best 80/20 Split Results")
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
    scores_5, average_5 = run_k_fold_df(X, y, num_folds1, best_k)
    print("KNN 5-Fold Cross Validation")
    print("Best k used:", best_k)
    print("Fold Accuracies:", scores_5)
    print("Average Accuracy:", average_5)
    print()

    # 10-fold cross validation using best k
    scores_10, average_10 = run_k_fold_df(X, y, num_folds2, best_k)
    print("KNN 10-Fold Cross Validation")
    print("Best k used:", best_k)
    print("Fold Accuracies:", scores_10)
    print("Average Accuracy:", average_10)
    print()

    # SVM 80/20 (Training/Test)
    y_test, predictions, metrics = svm_80_20(
        X,
        y,
        train_test_split_df,
        normalize_train_test_df,
        classification_metrics
    )

    print_svm_results("SVM 80/20 Results", metrics)

    print("Real Label  Predicted Label")
    for real, pred in zip(y_test, predictions):
        print(real, "       ", pred)

    print()

    # SVM 20/80 (Training/Test)
    y_test, predictions, metrics = svm_20_80(
        X,
        y,
        train_test_split_df,
        normalize_train_test_df,
        classification_metrics
    )

    print_svm_results("SVM 20/80 Results", metrics)

    # SVM 5-fold Cross Validation
    scores_5, metrics_5 = svm_k_fold(
        X,
        y,
        num_folds1,
        make_folds_df,
        get_train_test_from_folds_df,
        normalize_train_test_df,
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
        normalize_train_test_df,
        classification_metrics
    )

    print("SVM 10-Fold Cross Validation")
    print("Fold Accuracies:", scores_10)
    print_svm_results("Average 10-Fold Metrics", metrics_10)