import numpy as np
from sklearn.svm import SVC

# Train SVM and predict labels of test
# Trains with training data predicts class labels for test data
def svm_train_predict(X_train, y_train, X_test):
    svm = SVC(kernel='rbf')
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    return predictions.tolist()

# Run 80 training 20 testing with SVM
def svm_80_20(X, y, train_test_split_df, normalize_train_test_df, classification_metrics):
    X_train, y_train, X_test, y_test = train_test_split_df(X, y, train_percent=0.8)
    X_train, X_test = normalize_train_test_df(X_train, X_test)

    predictions = svm_train_predict(X_train, y_train, X_test)
    y_test_list = y_test.tolist()
    metrics = classification_metrics(y_test_list, predictions)

    return y_test_list, predictions, metrics

# Run 20 training 80 testing with SVM
def svm_20_80(X, y, train_test_split_df, normalize_train_test_df, classification_metrics):
    X_train, y_train, X_test, y_test = train_test_split_df(X, y, train_percent=0.2)
    X_train, X_test = normalize_train_test_df(X_train, X_test)

    predictions = svm_train_predict(X_train, y_train, X_test)
    y_test_list = y_test.tolist()
    metrics = classification_metrics(y_test_list, predictions)

    return y_test_list, predictions, metrics

 # Run K-fold cross validation with SVM
def svm_k_fold(
    X,
    y,
    num_folds,
    make_folds_df,
    get_train_test_from_folds_df,
    normalize_train_test_df,
    classification_metrics
):
    folds = make_folds_df(X, y, num_folds)

    scores = []
    metrics_list = []

    for i in range(num_folds):
        X_train, y_train, X_test, y_test = get_train_test_from_folds_df(folds, i)

        X_train, X_test = normalize_train_test_df(X_train, X_test)

        predictions = svm_train_predict(X_train, y_train, X_test)

        y_test_list = y_test.tolist()

        metrics = classification_metrics(y_test_list, predictions)

        scores.append(metrics["Accuracy"])
        metrics_list.append(metrics)

    avg_metrics = {
        "Accuracy": sum(m["Accuracy"] for m in metrics_list) / len(metrics_list),
        "Precision": sum(m["Precision"] for m in metrics_list) / len(metrics_list),
        "Sensitivity": sum(m["Sensitivity"] for m in metrics_list) / len(metrics_list),
        "Specificity": sum(m["Specificity"] for m in metrics_list) / len(metrics_list),
        "TP": sum(m["TP"] for m in metrics_list),
        "FP": sum(m["FP"] for m in metrics_list),
        "FN": sum(m["FN"] for m in metrics_list),
        "TN": sum(m["TN"] for m in metrics_list)
    }

    return scores, avg_metrics

# Prints SVM results in organized fashion
def print_svm_results(title, metrics):
    print(title)
    print("Accuracy:", metrics["Accuracy"])
    print("Precision:", metrics["Precision"])
    print("Sensitivity:", metrics["Sensitivity"])
    print("Specificity:", metrics["Specificity"])

    print("\nConfusion Matrix Values")
    print("TP:", metrics["TP"])
    print("FP:", metrics["FP"])
    print("FN:", metrics["FN"])
    print("TN:", metrics["TN"])
    print()