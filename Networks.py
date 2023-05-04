import tensorflow as tf

class ClassificationNetwork:
    def __init__(self, input_shape, learning_rate = 1e-3):
        self.model = tf.keras.Sequential()

        self.model.add(tf.keras.layers.Rescaling(1 / 255.0))
        self.model.add(tf.keras.layers.Conv2D(32, (16, 16), strides=(16, 16), activation='relu', input_shape=input_shape, name='conv_1'))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv_2'))
        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(100, activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.Dense(100, activation=tf.keras.activations.relu))

        self.model.add(tf.keras.layers.Dense(3))
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        
        self.input_shape = input_shape
        
    def get_V(self, state):
        return self.model.predict(state.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]), verbose=0)
    
    def fit(self, states, targets, epochs=2, batch_size=64, validation_split=0.2):
        return self.model.fit(x=states, y=targets, validation_split=validation_split, batch_size=batch_size, epochs=epochs)
    
    def save_weights(self, path):
        self.model.save_weights(path)
        
    def load_weights(self, path):
        self.model.load_weights(path)
        
class ValueNetwork:
    def __init__(self, input_shape, learning_rate = 1e-3):
        self.model = tf.keras.Sequential()

        self.model.add(tf.keras.layers.Rescaling(1 / 255.0))
        self.model.add(tf.keras.layers.Conv2D(32, (16, 16), strides=(16, 16), activation='relu', input_shape=input_shape, name='conv_1'))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv_2'))
        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(50, activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.Dense(50, activation=tf.keras.activations.relu))

        self.model.add(tf.keras.layers.Dense(1, activation=None))
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=[tf.keras.metrics.MeanSquaredError()])
        
        self.input_shape = input_shape
        
    def get_V(self, state):
        return self.model.predict(state.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]), verbose=0)
    
    def fit(self, states, targets, epochs=2, batch_size=64, validation_split=0.2):
        return self.model.fit(x=states, y=targets, validation_split=validation_split, batch_size=batch_size, epochs=epochs)
    
    def save_weights(self, path):
        self.model.save_weights(path)
        
    def load_weights(self, path):
        self.model.load_weights(path)