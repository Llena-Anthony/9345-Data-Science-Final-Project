"""
Description:
Defines a Decision Tree model builder.

This file only provides a function to sonstruct the model.
Traiining and evaluation are handled inside experiment scripts.

"""
from sklearn.tree import DecisionTreeClassifier

def build_decision_tree(max_depth=None):
    """
        Builds and returns a Decision Tree classifier.

        Parameters:
        - max_depth (int or None): Maximum tree depth (default=None)

        Returns:
        - DecisionTreeClassifier model
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion='gini',
        random_state=42,
    )
    return model