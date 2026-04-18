from sklearn.ensemble import RandomForestClassifier

def build_random_forest():
    """
    Create and return a Random Forest classifier for mineral classification.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=1,
        random_state=42,
        verbose=1
    )
    return model