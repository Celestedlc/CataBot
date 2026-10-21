#For first time users:
    #pip install tensorflow

import tensorflow as tf
import json

from tensorflow.keras import activations
from tensorflow.keras.optimizers import Optimizer

WORDS = {

        "hello": 0,
        "bye": 1
        }

RESPONSES = {
    
        0: "Hello, my name is Cata!",
        1: "Goodbye!"

        }
xTrain = tf.constant([0,1])
yTrain = tf.constant([0,1])
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(input_dim=2, output_dim=4),
    tf.keras.layers.Dense(2, activation = "softmax")

    ])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy")
model.fit(xTrain, yTrain, epochs = 200, verbose = 0)

tfLiteModel = tf.lite.TFLiteConverter.from_keras_model(model).convert()
with open("cataBot.tflite", "wb") as f:
    f.write(tfLiteModel)

with open("data.json", "w") as f:
    json.dump({"words": WORDS, "responses": RESPONSES}, f)
