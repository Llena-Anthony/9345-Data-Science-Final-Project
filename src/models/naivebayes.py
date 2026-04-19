from sklearn.naive_bayes import GaussianNB

def build_naive_bayes():
    """
    Create and return a Gaussian Naive Bayes classifier for mineral classification.
    """
    model = GaussianNB()
    return model