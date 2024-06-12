#Category + Scheme
import random
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents JSON file
with open("C:\\Users\\91875\\Desktop\\SchemeBot\\AI-Chatbot\\intents.json", "r") as file:
    intents = json.load(file)

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignore_words = ["?", "!"]

# Preprocess intents data
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize words in pattern
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize and normalize words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save preprocessed data
pickle.dump(words, open("C:\\Users\\91875\\Desktop\\SchemeBot\\AI-Chatbot\\words.pkl", "wb"))
pickle.dump(classes, open("C:\\Users\\91875\\Desktop\\SchemeBot\\AI-Chatbot\\classes.pkl", "wb"))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row, doc[1]])

random.shuffle(training)

train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])
train_category = np.array([item[2] for item in training])

# Define model architecture
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation="softmax"))

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save model
model.save("C:\\Users\\91875\\Desktop\\SchemeBot\\AI-Chatbot\\chatbot_model.h5", hist)
print("Model trained and saved.")







# Scheme Only

# import random
# import numpy as np
# import pickle
# import json
# import nltk
# from nltk.stem import WordNetLemmatizer
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import SGD

# # Download NLTK resources if not already downloaded
# nltk.download('punkt')
# nltk.download('wordnet')

# # Initialize WordNet lemmatizer
# lemmatizer = WordNetLemmatizer()

# # Load intents JSON file
# with open("C:\\Users\\91875\\Desktop\\SchemeBot\\AI-Chatbot\\intents.json", "r") as file:
#     intents = json.load(file)

# # Initialize lists for words, classes, and documents
# words = []
# classes = []
# documents = []
# ignore_words = ["?", "!"]

# # Preprocess intents data
# for intent in intents["intents"]:
#     for pattern in intent["patterns"]:
#         # Tokenize words in pattern
#         w = nltk.word_tokenize(pattern)
#         words.extend(w)
#         documents.append((w, intent["tag"]))
#         if intent["tag"] not in classes:
#             classes.append(intent["tag"])

# # Lemmatize and normalize words
# words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# words = sorted(list(set(words)))
# classes = sorted(list(set(classes)))

# # Save preprocessed data
# pickle.dump(words, open("C:\\Users\\91875\\Desktop\\SchemeBot\\AI-Chatbot\\words.pkl", "wb"))
# pickle.dump(classes, open("C:\\Users\\91875\\Desktop\\SchemeBot\\AI-Chatbot\\classes.pkl", "wb"))

# # Create training data
# training = []
# output_empty = [0] * len(classes)

# for doc in documents:
#     bag = []
#     pattern_words = doc[0]
#     pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
#     for w in words:
#         bag.append(1) if w in pattern_words else bag.append(0)
#     output_row = list(output_empty)
#     output_row[classes.index(doc[1])] = 1
#     training.append([bag, output_row])

# random.shuffle(training)

# train_x = [item[0] for item in training]
# train_y = [item[1] for item in training]

# # Convert to NumPy arrays
# train_x = np.array(train_x)
# train_y = np.array(train_y)

# # Define model architecture
# model = Sequential()
# model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation="softmax"))

# # Compile model
# sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# # Train model
# hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# # Save model
# model.save("C:\\Users\\91875\\Desktop\\SchemeBot\\AI-Chatbot\\chatbot_model.h5", hist)
# print("Model trained and saved.")
