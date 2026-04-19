from sklearn.ensemble import RandomForestClassifier

def build_random_forest():
    """
    Create and return a Random Forest classifier for mineral classification.
    """
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=1,
        random_state=42
    )
    return model