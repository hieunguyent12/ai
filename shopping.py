import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

TEST_SIZE = 0.4


def main():
    # load_data("shopping.csv")

    # Check command-line arguments
    # if len(sys.argv) != 2:
    #     sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data("shopping.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    print(sensitivity)
    print(specificity)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    labels = []
    evidence = []
    intItems = [
        "Administrative",
        "Informational",
        "ProductRelated",
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend",
        "Revenue",
    ]
    floatItems = [
        "Administrative_Duration",
        "Informational_Duration",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
    ]
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "June",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in reader:
            for i in intItems:
                if i == "Month":
                    row[i] = months.index(row[i])
                elif i == "VisitorType":
                    row[i] = int(row[i] == "Returning_Visitor")
                elif i == "Weekend" or i == "Revenue":
                    row[i] = int(row[i] == "TRUE")
                else:
                    row[i] = int(row[i])

            for i in floatItems:
                row[i] = float(row[i])

            vals = list(row.values())
            evidence.append(vals[:-1])
            labels.append(vals[-1])

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1, algorithm="brute")
    model.fit(evidence, labels)  # wtf does this fit method do?

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    sensitivity = 0
    specificity = 0

    positive = 0
    negative = 0
    totalPos = 0
    totalNeg = 0
    for index, actual in enumerate(labels):
        if actual == 1:
            totalPos += 1
            if actual == predictions[index]:
                positive += 1
        else:
            totalNeg += 1
            if actual == predictions[index]:
                negative += 1

    return (positive / totalPos, negative / totalNeg)


if __name__ == "__main__":
    main()
