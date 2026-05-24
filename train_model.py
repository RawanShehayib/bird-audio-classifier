import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load prepared data
X_train = np.load("X_train.npy")
X_test  = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

print("=== Training Random Forest... ===")


'''
Each of those 100 trees is literally a binary tree data structure.
Every internal node has exactly two children — the "yes" branch and the "no" branch. 
Every leaf is a terminal node with no children, storing a species label. 
Traversing from root to leaf to get a prediction is a standard tree traversal — O(depth) time complexity
'''
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    min_samples_leaf=2
)
model.fit(X_train, y_train) #more about that on read me
print("  Done!")

# Test on hidden clips : the 20% form prep data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n=== Accuracy: {accuracy * 100:.1f}% ===")

# Breakdown per species
print("\n=== Results per species ===")
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, "bird_model.pkl")
print("Model saved as bird_model.pkl")

'''problem faced: on the first test, the accuracy results: rcall for chaffinch is 0.58 and the accuracy for 
robin is 0.79, so we will need more data the model confuses chaffinch clips with robin most often.'''

'''problem faced again after loading more data sets: the accuracy of great tit dropped from 0.94 to 0.78
solution: capping every species at 155 clips regardless of how long the recordings are.'''