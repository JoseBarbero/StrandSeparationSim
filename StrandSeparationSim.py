import numpy as np
from ReadData import read_data_as_img
from Models import lenet, alexnet, widenet
from keras import backend as K
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    seed = 42

    data_dir = "../data"

    X, y = read_data_as_img(data_dir)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    np.random.seed(seed)

    K.set_image_data_format("channels_last")

    model = lenet()

    # Train the model
    model.fit(X_train, y_train, epochs=1, batch_size=128)

    # Evaluate model
    score = model.evaluate(X_test, y_test, batch_size=128)
    print("Test loss, Test accuracy", score)