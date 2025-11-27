
'''
Naive Bayes Classifier Own Implementation
'''

def read_data():
    x = []
    # first row is feature names, skip it
    with open("input_x.txt", "r") as file:
        for f in file:
            tab = f.strip().split(",")
            # check if first line is header
            if tab[0].isdigit():
                x.append([float(num) for num in tab])
            else:
                x.append([num for num in tab])
    y = []
    with open("input_y.txt", "r") as file:
        for f in file:
            tab = f.strip().split(",")
            if tab[0].isdigit():
                y.append(float(tab[0]))
            else:
                y.append(tab)
    return (x, y)
    
''' 
    Train the Naive Bayes algorithm 
    X: feature set
    Y: label set
    Returns a tuple containing the trained model and the prior probabilities for positive and negative classes.

    Bayes Theorem:
    if A and B are two events, then the conditional probability of A given B is given by:
    P(A|B) = (P(B|A) * P(A)) / P(B)
'''
def train_algorithm(X, Y):
    positive_possibility = sum(Y[1:]) / len(Y[1:len(Y)])
    negative_possibility = 1 - positive_possibility
    result = {}
    # per one feature which is in column
    for index,feature_names in enumerate(X[0]):
        # for each row
        for i in range(1, len(Y)):
            feature_positive_count = 0
            feature_negative_count = 0
            if Y[i] == 1 and X[i][index] == 1:
                feature_positive_count += 1
            elif Y[i] == 0 and X[i][index] == 1:
                feature_negative_count += 1
        feature_positive_possibility = feature_positive_count / sum(Y[1:])
        feature_negative_possibility = feature_negative_count / (len(Y[1:]) - sum(Y[1:]))
        result[feature_names] = {
            "positive": feature_positive_possibility,
            "negative": feature_negative_possibility
        }
    return (result, positive_possibility, negative_possibility)

''' 
    Test the Naive Bayes algorithm
    result: trained model
    positive_possibility: prior probability of positive class
    negative_possibility: prior probability of negative class
    X_test: feature set for testing
    Returns the predicted class label (1 for positive, 0 for negative).
    
'''

def test(result, positive_possibility, negative_possibility, X_test):
    positive_prob = positive_possibility
    negative_prob = negative_possibility
    for feature_index in range(len(X_test)):
        feature_value = X_test[feature_index]
        if feature_value == 1:
            positive_prob *= result[feature_index]["positive"]
            negative_prob *= result[feature_index]["negative"]
        else:
            positive_prob *= (1 - result[feature_index]["positive"])
            negative_prob *= (1 - result[feature_index]["negative"])
    if positive_prob > negative_prob:
        return 1
    else:
        return 0
    
if __name__ == "__main__":
    X, Y = read_data()
    print(X)
    print(Y)
    model = train_algorithm(X, Y)
    print(model)