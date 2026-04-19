from sklearn.neighbors import KNeighborsClassifier

def build_knn(n_neighbors=5):
    """
    Build and returns a KNN classifier.

    Parameters:
    - n_neigbors(int): Number of neighbors to use (default = 5).

    Returns:
    - KNeigborsClassifier model
    """

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights="distance", # better for imbalance
        metric="minkowski", #standard distance
        n_jobs=-1 # Use all cores
    )
    return model