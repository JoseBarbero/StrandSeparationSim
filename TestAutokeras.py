import numpy as np
import autokeras as ak
from ReadData import read_data_as_img, read_data_structured, read_data_st
from Preprocessing import ros, smote, adasyn
from sklearn.model_selection import train_test_split
import tensorflow as tf

# This solves a bug with RTX series that stops Cuda DNN from initializing correctly
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=config)


seed = 42

data_dir = "../data"

# # Sub345 ros
# X_train, y_train = read_data_as_img(data_dir, "OPNat(310|315|320|325|330|335|340).*train.*")
# X_val, y_val = read_data_as_img(data_dir, "OPNat(310|315|320|325|330|335|340).*val.*")
# X_test, y_test = read_data_as_img(data_dir, "OPNat.*hg17-test")
# new_X_train, new_y_train = ros(X_train, y_train)
# model = ak.ImageClassifier(overwrite=True, max_trials=5,project_name="sub345_ros",)
# model.fit(new_X_train, new_y_train, validation_data=(X_val, y_val), epochs=10)
# predicted_y = model.predict(X_test)
# print(model.evaluate(X_test, y_test))

# # All_temps ros
# X_train, y_train = read_data_as_img(data_dir, "OPNat.*train.*")
# X_val, y_val = read_data_as_img(data_dir, "OPNat.*val.*")
# X_test, y_test = read_data_as_img(data_dir, "OPNat.*hg17-test")
# new_X_train, new_y_train = ros(X_train, y_train)
# model = ak.ImageClassifier(overwrite=True, max_trials=5,project_name="all_temps_ros",)
# model.fit(new_X_train, new_y_train, validation_data=(X_val, y_val), epochs=10)
# predicted_y = model.predict(X_test)
# print(model.evaluate(X_test, y_test))

# # Sub345 smote
# X_train, y_train = read_data_as_img(data_dir, "OPNat(310|315|320|325|330|335|340).*train.*")
# X_val, y_val = read_data_as_img(data_dir, "OPNat(310|315|320|325|330|335|340).*val.*")
# X_test, y_test = read_data_as_img(data_dir, "OPNat.*hg17-test")
# new_X_train, new_y_train = smote(X_train, y_train)
# model = ak.ImageClassifier(overwrite=True, max_trials=5,project_name="sub345_smote",)
# model.fit(new_X_train, new_y_train, validation_data=(X_val, y_val), epochs=10)
# predicted_y = model.predict(X_test)
# print(model.evaluate(X_test, y_test))

# # All_temps smote
# X_train, y_train = read_data_as_img(data_dir, "OPNat.*train.*")
# X_val, y_val = read_data_as_img(data_dir, "OPNat.*val.*")
# X_test, y_test = read_data_as_img(data_dir, "OPNat.*hg17-test")
# new_X_train, new_y_train = smote(X_train, y_train)
# model = ak.ImageClassifier(overwrite=True, max_trials=5,project_name="all_temps_smote",)
# model.fit(new_X_train, new_y_train, validation_data=(X_val, y_val), epochs=10)
# predicted_y = model.predict(X_test)
# print(model.evaluate(X_test, y_test))

# # Sub345 adasyn
# X_train, y_train = read_data_as_img(data_dir, "OPNat(310|315|320|325|330|335|340).*train.*")
# X_val, y_val = read_data_as_img(data_dir, "OPNat(310|315|320|325|330|335|340).*val.*")
# X_test, y_test = read_data_as_img(data_dir, "OPNat.*hg17-test")
# new_X_train, new_y_train = adasyn(X_train, y_train)
# model = ak.ImageClassifier(overwrite=True, max_trials=5,project_name="sub345_adasyn",)
# model.fit(new_X_train, new_y_train, validation_data=(X_val, y_val), epochs=10)
# predicted_y = model.predict(X_test)
# print(model.evaluate(X_test, y_test))

# # All_temps adasyn
# X_train, y_train = read_data_as_img(data_dir, "OPNat.*train.*")
# X_val, y_val = read_data_as_img(data_dir, "OPNat.*val.*")
# X_test, y_test = read_data_as_img(data_dir, "OPNat.*hg17-test")
# new_X_train, new_y_train = adasyn(X_train, y_train)
# model = ak.ImageClassifier(overwrite=True, max_trials=5,project_name="all_temps_adasyn",)
# model.fit(new_X_train, new_y_train, validation_data=(X_val, y_val), epochs=10)
# predicted_y = model.predict(X_test)
# print(model.evaluate(X_test, y_test))

# Structured
# X_train, y_train = read_data_structured(data_dir, "OPNat.*train.*")
# X_val, y_val = read_data_structured(data_dir, "OPNat.*val.*")
# X_test, y_test = read_data_structured(data_dir, "OPNat.*test.*")
# model = ak.ImageClassifier(overwrite=True, max_trials=5, project_name="structured",)
# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
# predicted_y = model.predict(X_test)
# print(model.evaluate(X_test, y_test))

# All temps stacking temperatures
X_train, y_train = read_data_st(data_dir, "train")
X_val, y_val = read_data_st(data_dir, "val")
X_test, y_test = read_data_st(data_dir, "test")
new_X_train, new_y_train = smote(X_train, y_train)
model = ak.ImageClassifier(overwrite=True, max_trials=5, project_name="st_smote")
model.fit(new_X_train, new_y_train, validation_data=(X_val, y_val), epochs=10)
predicted_y = model.predict(X_test)
print(model.evaluate(X_test, y_test))

# # < 345 stacking temperatures
# X_train, y_train = read_data_st(data_dir, "train", [range(310, 350, 5)])
# X_val, y_val = read_data_st(data_dir, "val", [range(310, 350, 5)])
# X_test, y_test = read_data_st(data_dir, "test", [range(310, 350, 5)])
# model = ak.ImageClassifier(overwrite=True, max_trials=5,project_name="st_sub345",)
# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
# predicted_y = model.predict(X_test)
# print(model.evaluate(X_test, y_test))