import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train = pd.read_csv('./data/new_data/X_train_model_1.csv')
y_train = pd.read_csv('./data/new_data/y_train_model_1.csv')
X_test = pd.read_csv('./data/new_data/X_test_model_1.csv')
y_test = pd.read_csv('./data/new_data/y_test_model_1.csv')

tree_model = DecisionTreeClassifier(random_state=189)
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('accuracy:', acc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Tree Depth:", tree_model.get_depth())
print("Number of Leaves:", tree_model.get_n_leaves())
