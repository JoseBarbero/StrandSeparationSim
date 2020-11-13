import numpy as np
import autokeras as ak
from ReadData import read_data_as_img
from sklearn.model_selection import train_test_split

seed = 42

data_dir = "../data"

X, y = read_data_as_img(data_dir)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)


# It tries 10 different models.
clf = ak.ImageClassifier(overwrite=True, max_trials=5)
# Feed the structured data classifier with training data.
clf.fit(X_train, y_train, epochs=3)
# Predict with the best model.
predicted_y = clf.predict(X_test)
# Evaluate the best model with testing data.
print(clf.evaluate(X_test, y_test))