from _operator import truediv


# Given the predictions and labels, this calculates the accuracy, recall, precision and f1.
def calc_metrics(predictions, labels):

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = round(accuracy_score(labels, predictions), 4)

    recall = round(recall_score(labels, predictions,
                   average='weighted', zero_division=0), 4)
    precision = round(precision_score(labels, predictions,
                      average='weighted', zero_division=0), 4)

    f1 = round(f1_score(labels, predictions,
               average='weighted', zero_division=0), 4)

    return accuracy, recall, precision, f1

# Given the predictions and labels, this function calculates the confusion matrix.
def calc_confusion(predictions, labels):
    from sklearn.metrics import confusion_matrix
    conf = confusion_matrix(labels, predictions)
    return conf


# TODO: Soham
def calc_auc():
    pass

# to test differences in results of two models
# return F and p value. if the p-value is smaller than alpha (typically is 0.05) then the results
# deemed to be statistically differ
# http://rasbt.github.io/mlxtend/user_guide/evaluate/ftest/


def f_test(models, labels):
    from mlxtend.evaluate import ftest

    f, p_value = ftest(labels,
                       models[0], models[1])
    return f, p_value
