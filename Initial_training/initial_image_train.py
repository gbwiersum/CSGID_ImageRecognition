# Much of the code used here is based on: https://keras.io/examples/keras_recipes/tfrecord/
import tensorflow as tf
from utils.DataGenerator import get_dataset
import glob

train_list = glob.glob('../train-jpg/*')
test_list = glob.glob('../test-jpg/*')
valid_list = test_list

# Tuning and training params:
BATCH_SIZE = 1
IMAGE_SIZE = [299, 299, 3]
keras_batch_size = BATCH_SIZE
learning_rate = 0.095
epochs = 1

print("Train Files: ", len(train_list))
print("Test Files: ", len(test_list))

train_dataset = get_dataset(train_list, batch_size=BATCH_SIZE, im_size=IMAGE_SIZE)
valid_dataset = get_dataset(test_list, batch_size=BATCH_SIZE, im_size=IMAGE_SIZE)

model = tf.keras.applications.InceptionV3(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=4,
    classifier_activation="softmax",
)

model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy,
              metrics=tf.keras.metrics.categorical_accuracy)

model.fit(train_dataset, validation_data=valid_dataset, batch_size=keras_batch_size, epochs=epochs)
model.save('../Models/Baseline-Model.h5')
