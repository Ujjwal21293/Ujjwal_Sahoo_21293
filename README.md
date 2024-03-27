Question 3
import tensorflow as tf
from tensorflow.keras import layers, models

def model1(input_shape, num_classes):
    model=models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))  
    model.add(layers.Dense(128, activation='relu'))     
    model.add(layers.Dense(128, activation='relu'))     
    model.add(layers.Dense(128, activation='relu'))     
    model.add(layers.Dense(num_classes, activation='softmax')) 
    return model

def preprocess(images):
    images=images / 255.0  
    return images

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.cifar10.load_data()
x_train, x_test=preprocess(x_train), preprocess(x_test)

input_shape=x_train.shape[1:]
num_classes=len(set(y_train.flatten())) 
learning_rate=0.001
batch=64
epochs=20

model=model1(input_shape, num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(x_train, y_train, batch_size=batch, epochs=epochs,
                    validation_data=(x_test, y_test))

test_loss, test_accuracy=model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
