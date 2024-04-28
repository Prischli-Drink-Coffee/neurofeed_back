import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.layers import Embedding, BatchNormalization, Concatenate
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

max_words = 20000
max_len = 512
embedding_dim = 32

# Branch 1
branch1 = Sequential()
branch1.add(Embedding(max_words, embedding_dim, input_length=max_len))
branch1.add(Conv1D(512, 3, padding='same', activation='relu'))
branch1.add(BatchNormalization())
branch1.add(ReLU())
branch1.add(Dropout(0.5))
branch1.add(GlobalMaxPooling1D())

# Branch 2
branch2 = Sequential()
branch2.add(Embedding(max_words, embedding_dim, input_length=max_len))
branch2.add(Conv1D(512, 3, padding='same', activation='relu'))
branch2.add(BatchNormalization())
branch2.add(ReLU())
branch2.add(Dropout(0.5))
branch2.add(GlobalMaxPooling1D())

concatenated = Concatenate()([branch1.output, branch2.output])

hid_layer = Dense(512, activation='relu')(concatenated)
dropout = Dropout(0.3)(hid_layer)
output_layer = Dense(2, activation='sigmoid')(dropout)

model = Model(inputs=[branch1.input, branch2.input], outputs=output_layer)


import numpy as np

test_data = pd.read_csv('C:/Users/NightMare/Desktop/neurofeed_back/test_df.csv')

# Функция predict остается неизменной, но при вызове используется numpy.argmax для определения предсказанного класса

def predict(text, model_path, token_path):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import pickle
    from tensorflow.keras.models import load_model
    
    model = load_model(model_path)
    
    with open(token_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    sequences = tokenizer.texts_to_sequences(text)
    x_new = pad_sequences(sequences, maxlen=512)
    predictions = model.predict([x_new, x_new])
    
    # Использование numpy.argmax для определения предсказанного класса
    predicted_labels = np.argmax(predictions, axis=1)
    
    return predicted_labels

# Загрузка данных test_data и их предсказание
texts = list(test_data['text'])
labels = list(test_data['label'])

predicted_labels = predict(texts, 'nlp_np.h5', 'tokenizer_np.pkl')

df = pd.DataFrame()
df = test_data
df['predict_np'] = predicted_labels
# Переименование столбца "labels" в "label"
df.rename(columns={"label": "label_np"}, inplace=True)
df

