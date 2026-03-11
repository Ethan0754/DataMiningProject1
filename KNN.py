import numpy as np
import random
import pandas as pd
import itertools


def pre_processing(df, target_col='Purchase_Intent'):
    df = df.copy()

    # Drop identifier-like columns only
    drop_cols = ['Customer_ID', 'Location', 'Gender', 'Marital_Status', 'Purchase_Category', 'Purchase_Channel', 'Device_Used_for_Shopping', 'Payment_Method', 'Time_of_Purchase', 'Purchase_Intent', 'Shipping_Preference']
    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    df.drop(columns=existing_drop_cols, inplace=True)

    # Clean Purchase_Amount
    if 'Purchase_Amount' in df.columns:
        df['Purchase_Amount'] = (
            df['Purchase_Amount']
            .replace(r'[\$, ]', '', regex=True)
            .astype(float)
        )

    # Convert boolean columns to binary
    bool_cols = ['Discount_Used', 'Customer_Loyalty_Program_Member']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Ordinal encoding
    ordinal_mappings = {
        'Income_Level': {
            'Low': 0,
            'Middle': 1,
            'High': 2
        },
        'Education_Level': {
            'High School': 0,
            "Bachelor's": 1,
            "Master's": 2,
            'PhD': 3
        },
        'Social_Media_Influence': {
            'None': 0,
            'Low': 1,
            'Medium': 2,
            'High': 3
        },
        'Discount_Sensitivity': {
            'Not Sensitive': 0,
            'Somewhat Sensitive': 1,
            'Very Sensitive': 2
        },
        'Engagement_with_Ads': {
            'None': 0,
            'Low': 1,
            'Medium': 2,
            'High': 3
        },
        'Occupation': {
            'Middle': 0,
            'High': 1
        }
    }

    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Convert numeric columns
    numeric_cols = [
        'Age',
        'Frequency_of_Purchase',
        'Brand_Loyalty',
        'Product_Rating',
        'Time_Spent_on_Product_Research(hours)',
        'Return_Rate',
        'Customer_Satisfaction',
        'Time_to_Decision'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle date column
    if 'Time_of_Purchase' in df.columns:
        df['Time_of_Purchase'] = pd.to_datetime(df['Time_of_Purchase'], errors='coerce')
        df['Purchase_Month'] = df['Time_of_Purchase'].dt.month
        df['Purchase_Day'] = df['Time_of_Purchase'].dt.day
        df['Purchase_DayOfWeek'] = df['Time_of_Purchase'].dt.dayofweek
        df.drop(columns=['Time_of_Purchase'], inplace=True)

    # Separate target before encoding predictors
    y = None
    if target_col in df.columns:
        y = df[target_col].copy()
        df.drop(columns=[target_col], inplace=True)

    # One-hot encode remaining nominal columns
    nominal_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    if nominal_cols:
        df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)

    # Fill missing values
    df = df.fillna(0)

    # Convert everything to float
    df = df.astype(float)

    # Encode target separately
    if y is not None:
        y = pd.Categorical(y).codes

    return df, y

def train_test_split_df(X, y, train_percent=0.8):
    data = X.copy()
    data['target'] = y

    #data = data.sample(frac=1, random_state=42).reset_index(drop=True)

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
    # find the minimum value in each column of the training data
    mins = X_train.min()

    # find the maximum value in each column of the training data
    maxs = X_train.max()

    # find the range for each column
    ranges = maxs - mins

    # if range is 0, set it to 1 to avoid divide by zero
    ranges[ranges == 0] = 1

    # normalize training data using min-max normalization
    X_train_norm = (X_train - mins) / ranges

    # normalize test data using the training mins and ranges
    X_test_norm = (X_test - mins) / ranges

    return X_train_norm, X_test_norm


def dataframe_to_lists(X, y):
    X_values = X.to_numpy(dtype=float).tolist()
    y_values = y.tolist()

    return X_values, y_values


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


def make_folds_df(X, y, k):
    data = X.copy()
    data['target'] = y

    #data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    folds = []
    fold_size = len(data) // k #floor division

    #isolate the folds into their own index of a list (folds)
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
        if y_true[i] == y_pred[i]: #Count the amount of correct predictions
            correct += 1

    return correct / len(y_true) #Find correctness percentage


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


if __name__ == "__main__":
    # set the random seed so shuffling happens the same way every run
    random.seed(42)

    df = pd.read_csv(
        "Group 5-Ecommerce_Consumer_Behavior_Analysis_Data.csv",
        keep_default_na=False
    )

    df = df[['Purchase_Amount', 'Discount_Sensitivity', 'Discount_Used']] #Restrict columns

    X, y = pre_processing(df, 'Discount_Used')

    k = 7
    num_folds1 = 5
    num_folds2 = 10

    # 80/20 split
    X_train, y_train, X_test, y_test = train_test_split_df(X, y, train_percent=0.8)

    # normalize training and test data
    X_train, X_test = normalize_train_test_df(X_train, X_test)

    # convert dataframes to lists so KNN functions can use them
    X_train_list, y_train_list = dataframe_to_lists(X_train, y_train)
    X_test_list, y_test_list = dataframe_to_lists(X_test, y_test)

    predictions = predict_all(X_train_list, y_train_list, X_test_list, k)
    accuracy = accuracy_score(y_test_list, predictions)

    print("80/20 Split Results")
    print("Predictions:", predictions)
    print("Actual:", y_test_list)
    print("Accuracy:", accuracy)
    print()

    # 5-fold cross validation
    scores_5, average_5 = run_k_fold_df(X, y, num_folds1, k)
    print("5-Fold Cross Validation")
    print("Fold Accuracies:", scores_5)
    print("Average Accuracy:", average_5)
    print()

    # 10-fold cross validation
    scores_10, average_10 = run_k_fold_df(X, y, num_folds2, k)
    print("10-Fold Cross Validation")
    print("Fold Accuracies:", scores_10)
    print("Average Accuracy:", average_10)