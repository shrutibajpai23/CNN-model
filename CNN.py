import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 1. Tokenize the text
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(["I am Shruti Bajpai", "I am undergraduate"])
print("Word Index:", tokenizer.word_index)

# 2. Convert new text to sequences
sequences = tokenizer.texts_to_sequences(["I am Shruti Bajpai"])
print("Sequence:", sequences)

# 3. Pad the sequences
padded = pad_sequences(sequences, padding='post')
print("Padded Sequence:", padded)

# 4. Build a model with embedding layer
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=64, input_length=padded.shape[1]),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 5. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Create dummy training data
x_train = np.random.randint(1000, size=(1000, padded.shape[1]))
y_train = np.random.randint(2, size=(1000, 1))

# 7. Train the model
model.fit(x_train, y_train, epochs=5)