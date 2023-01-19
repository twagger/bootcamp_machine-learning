"""
Metrics to help evaluate classifier.

When calculating performance scores for a multiclass classifier, we like to
compute a separate score for each class that your classifier learned to
discriminate (in a one-vs-all manner). In other words, for agiven Class A,
we want a score that can answer the question: "how good is the model at
assigning A objects to Class A, and at NOT assigning non-A objects to Class A?"

False positive: when a non-A object is assigned to Class A.
False negative: when an A object is assigned to another class than Class A.

"""
import sys
import numpy as np


# Helper function
def tf_metrics(y: np.ndarray, y_hat: np.ndarray, pos_label=None) -> tuple:
    """
    Returns as a tuple in that order :
        true positive number
        false positive number
        true negative number
        false negative number
    """
    # type tests
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)
                or (pos_label is not None and (not isinstance(pos_label, int)
                    and not isinstance(pos_label, str)))):
        print('Something went wrong', file=sys.stderr)
        return None
    # shape test
    if y.shape != y_hat.shape:
        print('Something went wrong', file=sys.stderr)
        return None
    # initialize variables
    tp, fp, tn, fn = [0]*4
    # if global for all classes
    if pos_label==None:
        # loop on every class in the original y vector
        for class_ in np.unique(y):
            tp += np.sum(np.logical_and(y_hat == class_, y == class_))
            fp += np.sum(np.logical_and(y_hat == class_, y != class_))
            tn += np.sum(np.logical_and(y_hat != class_, y != class_))
            fn += np.sum(np.logical_and(y_hat != class_, y == class_))
    else: # focus on one class
        tp += np.sum(np.logical_and(y_hat == pos_label, y == pos_label))
        fp += np.sum(np.logical_and(y_hat == pos_label, y != pos_label))
        tn += np.sum(np.logical_and(y_hat != pos_label, y != pos_label))
        fn += np.sum(np.logical_and(y_hat != pos_label, y == pos_label))

    return (tp, fp, tn, fn)


def accuracy_score_(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Compute the accuracy score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
    Returns:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # type tests
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            print('Something went wrong', file=sys.stderr)
            return None
        # shape test
        if y.shape != y_hat.shape:
            print('Something went wrong', file=sys.stderr)
            return None
        # calculation
        tp, fp, tn, fn = tf_metrics(y, y_hat)
        return (tp + tn) / (tp + fp + tn + fn)

    except (TypeError, ValueError, AttributeError) as exc:
        print(exc, file=sys.stderr)
        return None

def precision_score_(y: np.ndarray, y_hat: np.ndarray,
                     pos_label: int = 1) -> float:
    """
    Compute the precision score : model's ability to not classify positive
                                  examples as negative.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score
            (default=1)
    Return:
        The precision score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # type tests
        if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)
                or (not isinstance(pos_label, int) 
                    and not isinstance(pos_label, str))):
            print('Something went wrong', file=sys.stderr)
            return None
        # shape test
        if y.shape != y_hat.shape:
            print('Something went wrong', file=sys.stderr)
            return None
        # calculation
        tp, fp, _, _ = tf_metrics(y, y_hat, pos_label)
        return tp / (tp + fp)

    except (TypeError, ValueError, AttributeError) as exc:
        print(exc, file=sys.stderr)
        return None


def recall_score_(y: np.ndarray, y_hat: np.ndarray,
                  pos_label: int = 1) -> float:
    """
    Compute the recall score : model's ability to detect positive examples.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score
            (default=1)
    Return:
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # type tests
        if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)
                or (not isinstance(pos_label, int) 
                    and not isinstance(pos_label, str))):
            print('Something went wrong', file=sys.stderr)
            return None
        # shape test
        if y.shape != y_hat.shape:
            print('Something went wrong', file=sys.stderr)
            return None
        # calculation
        tp, _, _, fn = tf_metrics(y, y_hat, pos_label)
        return tp / (tp + fn)

    except (TypeError, ValueError, AttributeError) as exc:
        print(exc, file=sys.stderr)
        return None


def f1_score_(y: np.ndarray, y_hat: np.ndarray,
              pos_label: int = 1) -> float:
    """
    Compute the f1 score : harmonic mean of precision and recall. often used
                           for imbalanced datasets where it is important to
                           minimize false negatives while minimizing false
                           positives.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score
            (default=1)
    Returns:
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # type tests
        if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)
                or (not isinstance(pos_label, int) 
                    and not isinstance(pos_label, str))):
            print('Something went wrong', file=sys.stderr)
            return None
        # shape test
        if y.shape != y_hat.shape:
            print('Something went wrong', file=sys.stderr)
            return None
        # calculation
        precision = precision_score_(y, y_hat, pos_label=pos_label)
        recall = recall_score_(y, y_hat, pos_label=pos_label)
        return (2 * precision * recall) / (precision + recall)

    except (TypeError, ValueError, AttributeError) as exc:
        print(exc, file=sys.stderr)
        return None


if __name__ == "__main__":

    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, \
                                recall_score, f1_score

    # Example 1:
    # -------------------------------------------------------------------------
    y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))

    print("Example 1")
    # Accuracy
    ## your implementation
    print(accuracy_score_(y, y_hat)) # -> 0.5
    ## sklearn implementation
    print(accuracy_score(y, y_hat)) # -> 0.5

    # Precision
    ## your implementation
    print(precision_score_(y, y_hat)) # -> 0.4
    ## sklearn implementation
    print(precision_score(y, y_hat)) # -> 0.4

    # Recall
    ## your implementation
    print(recall_score_(y, y_hat)) # -> 0.6666666666666666
    ## sklearn implementation
    print(recall_score(y, y_hat)) # -> 0.6666666666666666

    # F1-score
    ## your implementation
    print(f1_score_(y, y_hat)) # -> 0.5
    ## sklearn implementation
    print(f1_score(y, y_hat)) # -> 0.5

    # Example 2:
    # -------------------------------------------------------------------------
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog',
                      'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet',
                  'dog', 'norminet'])

    print("Example 2")
    # Accuracy
    ## your implementation
    print(accuracy_score_(y, y_hat)) # -> 0.625
    ## sklearn implementation
    print(accuracy_score(y, y_hat)) # -> 0.625

    # Precision
    ## your implementation
    print(precision_score_(y, y_hat, pos_label='dog')) # -> 0.6
    ## sklearn implementation
    print(precision_score(y, y_hat, pos_label='dog')) # -> 0.6

    # Recall
    ## your implementation
    print(recall_score_(y, y_hat, pos_label='dog')) # -> 0.75
    ## sklearn implementation
    print(recall_score(y, y_hat, pos_label='dog')) # -> 0.75

    # F1-score
    ## your implementation
    print(f1_score_(y, y_hat, pos_label='dog')) # -> 0.6666666666666665
    ## sklearn implementation
    print(f1_score(y, y_hat, pos_label='dog')) # -> 0.6666666666666665


    # Example 3:
    # -------------------------------------------------------------------------
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog',
                      'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet',
                  'dog', 'norminet'])

    print("Example 3")
    # Precision
    ## your implementation
    print(precision_score_(y, y_hat, pos_label='norminet')) # -> 0.666666666...
    ## sklearn implementation
    print(precision_score(y, y_hat, pos_label='norminet')) # -> 0.6666666666...

    # Recall
    ## your implementation
    print(recall_score_(y, y_hat, pos_label='norminet')) # -> 0.5
    ## sklearn implementation
    print(recall_score(y, y_hat, pos_label='norminet')) # -> 0.5

    # F1-score
    ## your implementation
    print(f1_score_(y, y_hat, pos_label='norminet')) # -> 0.5714285714285715
    ## sklearn implementation
    print(f1_score(y, y_hat, pos_label='norminet')) # -> 0.5714285714285715
