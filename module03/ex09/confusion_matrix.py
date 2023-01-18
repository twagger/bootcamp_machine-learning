"""
Confusion matrix.

A confusion matrix is a commonly used tool to evaluate the performance of a
classification model in machine learning. It allows to visualize the errors
made by the model by comparing the predicted values to the actual values. The
confusion matrix is usually presented in the form of a table, where the rows
represent the actual values and the columns represent the predicted values.
The cells contain the number of occurrences where the model predicted a
certain value for a certain actual class. Therefore, the confusion matrix
allows to understand the errors made by the model, and to determine the areas
in which improvement is necessary.
"""
import sys
import os
import numpy as np
import pandas as pd


def confusion_matrix_(y_true: np.ndarray, y_hat: np.ndarray,
                      labels: list = None,
                      df_option: bool = False) -> np.ndarray:
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        labels: optional, a list of labels to index the matrix.
                This may be used to reorder or select a subset of labels.
                (default=None)
        df_option: optional, if set to True the function will return a pandas
                   DataFrame instead of a numpy array. (default=False)
    Return:
        The confusion matrix as a numpy array or a pandas DataFrame according
            to df_option value.
        None if any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # type tests
        if (not isinstance(y_true, np.ndarray)
                or not isinstance(y_hat, np.ndarray)
                or (not isinstance(labels, list) and labels is not None)
                or not isinstance(df_option, bool)):
            print('Something went wrong', file=sys.stderr)
            return None
        # shape test
        if y_true.shape != y_hat.shape:
            print('Something went wrong', file=sys.stderr)
            return None
        # calculation
        # labels
        if labels is None:
            temp_all = np.c_[y_true, y_hat]
            labels = np.unique(temp_all)
        # loop
        matrix = [np.sum(np.logical_and(y == label, y_hat == lab))
                  for label in labels for lab in labels]
        matrix = np.array(matrix).reshape(-1, len(labels))
        result = matrix
        # df option
        if df_option is True:
            del result
            result = pd.DataFrame(data = matrix, index = labels,
                                  columns = labels)
        return result

    except (TypeError, ValueError, AttributeError) as exc:
        print(exc, file=sys.stderr)
        return None


if __name__ == "__main__":

    import numpy as np
    from sklearn.metrics import confusion_matrix
    y_hat = np.array([['norminet'], ['dog'], ['norminet'], ['norminet'],
                      ['dog'], ['bird']])
    y = np.array([['dog'], ['dog'], ['norminet'], ['norminet'], ['dog'],
                  ['norminet']])

    # Example 1:
    ## your implementation
    print(confusion_matrix_(y, y_hat))
    ## Output: array([[0 0 0]
    #                 [0 2 1]
    #                 [1 0 2]])

    ## sklearn implementation
    print(confusion_matrix(y, y_hat))
    ## Output: array([[0 0 0]
    #                 [0 2 1]
    #                 [1 0 2]])

    # Example 2:
    ## your implementation
    print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))
    ## Output: array([[2 1]
    #                 [0 2]])

    ## sklearn implementation
    print(confusion_matrix(y, y_hat, labels=['dog', 'norminet']))
    ## Output: array([[2 1]
    #                 [0 2]])

    #Example 3:
    print(confusion_matrix_(y, y_hat, df_option=True))
    #Output:
    #          bird dog norminet
    # bird        0   0        0
    # dog         0   2        1
    # norminet    1   0        2

    #Example 2:
    print(confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True))
    #Output:
    #      bird dog
    # bird    0   0
    # dog     0   2
