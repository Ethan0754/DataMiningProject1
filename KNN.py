import numpy as np
import random

dataset = [
    [0.12, 0.18, 0.22, 0.15, 0.10, 0],
    [0.20, 0.25, 0.30, 0.22, 0.18, 0],
    [0.18, 0.10, 0.28, 0.20, 0.14, 0],
    [0.82, 0.78, 0.88, 0.80, 0.76, 1],
    [0.75, 0.85, 0.79, 0.83, 0.81, 1],
    [0.88, 0.72, 0.84, 0.77, 0.79, 1],
    [0.35, 0.40, 0.45, 0.38, 0.42, 0],
    [0.40, 0.32, 0.50, 0.36, 0.39, 0],
    [0.58, 0.62, 0.55, 0.60, 0.57, 1],
    [0.63, 0.57, 0.60, 0.64, 0.61, 1],
    [0.28, 0.30, 0.25, 0.27, 0.29, 0],
    [0.70, 0.68, 0.73, 0.71, 0.69, 1],
    [0.16, 0.20, 0.26, 0.18, 0.15, 0],
    [0.80, 0.76, 0.85, 0.79, 0.78, 1],
    [0.38, 0.35, 0.47, 0.37, 0.40, 0],
    [0.60, 0.59, 0.58, 0.62, 0.60, 1]
]


def euclidean_distance(train_data, test_data):
    squared_distance = 0
    for i in range(len(train_data)):
        squared_distance += np.square(train_data[i] - test_data[i]) #under the square root
    distance = np.sqrt(squared_distance) #calculating the square root
    return distance


def k_nearest_neighbors(X_train, y_train, test_point, k):
    distances = []

    #find the distance from each point in x_train to test_point
    for i in range(len(X_train)):
        distance = euclidean_distance(X_train[i], test_point)
        distances.append((distance, y_train[i]))

    distances.sort(key=lambda pair: pair[0]) #sort by the first index in tuple

    neighbor_labels = [] #assign the labels from k nearest neighbors
    for i in range(k):
        neighbor_labels.append(distances[i][1])

    return neighbor_labels


def predict_class(neighbor_labels):
    counts = {}

    for label in neighbor_labels: #count how many points are for one label and the other
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1

    best_label = None
    best_count = -1

    for label in counts: #Find which label has more points than the other
        if counts[label] > best_count:
            best_count = counts[label]
            best_label = label

    return best_label


def predict_all(X_train, y_train, X_test, k):
    predictions = []

    for test_point in X_test: #using test data, return the predictions for each line in x_test
        neighbor_labels = k_nearest_neighbors(X_train, y_train, test_point, k)
        prediction = predict_class(neighbor_labels)
        predictions.append(prediction)

    return predictions


def train_test_split(dataset, train_percent=0.8):
    data = dataset[:]  # creates copy
    random.shuffle(data) #shuffle the dataset

    # split the data by training_percent
    split_index = int(len(data) * train_percent)
    train_data = data[:split_index]
    test_data = data[split_index:]

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for row in train_data:
        X_train.append(row[:-1]) #all but the last column (the target label)
        y_train.append(row[-1]) #the target label column

    for row in test_data:
        X_test.append(row[:-1]) #all but the last column (the target label)
        y_test.append(row[-1]) #the target label column

    return X_train, y_train, X_test, y_test


def make_folds(dataset, k):
    data = dataset[:] #creates copy
    random.shuffle(data)

    folds = []
    fold_size = len(data) // k #floor division

    #isolate the folds into their own index of a list (folds)
    start = 0
    for i in range(k):
        if i == k - 1:
            fold = data[start:]
        else:
            fold = data[start:start + fold_size]

        folds.append(fold)
        start += fold_size

    return folds


def get_train_test_from_folds(folds, test_fold_index):
    """Function is the same as train_test_split, but for folds instead of 80/20 split"""
    train_data = []
    test_data = folds[test_fold_index]

    for i in range(len(folds)):
        if i != test_fold_index:
            train_data.extend(folds[i])

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for row in train_data:
        X_train.append(row[:-1])
        y_train.append(row[-1])

    for row in test_data:
        X_test.append(row[:-1])
        y_test.append(row[-1])

    return X_train, y_train, X_test, y_test


def accuracy_score(y_true, y_pred):
    correct = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]: #Count the amount of correct predictions
            correct += 1

    return correct / len(y_true) #Find correctness percentage


def run_k_fold(dataset, num_folds, k):
    """Same as predict_all but for folds"""
    folds = make_folds(dataset, num_folds) #returns folds as triple nested list
    scores = []

    for i in range(num_folds):
        X_train, y_train, X_test, y_test = get_train_test_from_folds(folds, i)

        predictions = predict_all(X_train, y_train, X_test, k)
        accuracy = accuracy_score(y_test, predictions)
        scores.append(accuracy)

    average_accuracy = sum(scores) / len(scores)
    return scores, average_accuracy


if __name__ == "__main__":
    k = 3
    num_folds1 = 5
    num_folds2 = 10

    # 80/20 split
    X_train, y_train, X_test, y_test = train_test_split(dataset, train_percent=0.8)
    predictions = predict_all(X_train, y_train, X_test, k)
    accuracy = accuracy_score(y_test, predictions)

    print("80/20 Split Results")
    print("Predictions:", predictions)
    print("Actual:", y_test)
    print("Accuracy:", accuracy)
    print()

    # 5-fold cross validation
    scores_5, average_5 = run_k_fold(dataset, num_folds1, k)
    print("5-Fold Cross Validation")
    print("Fold Accuracies:", scores_5)
    print("Average Accuracy:", average_5)
    print()

    # 10-fold cross validation
    scores_10, average_10 = run_k_fold(dataset, num_folds2, k)
    print("10-Fold Cross Validation")
    print("Fold Accuracies:", scores_10)
    print("Average Accuracy:", average_10)