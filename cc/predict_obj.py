import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.layers import Embedding, BatchNormalization, Concatenate
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


def predict_obj(str_array: list) -> int:

    def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
        epsilon = 1e-9
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_loss = -alpha * (1 - pt) ** gamma * tf.math.log(pt)
        return tf.reduce_mean(focal_loss)

    def f1_score(y_true, y_pred):
        # Округляем предсказанные значения до бинарных (0 или 1)
        y_pred = K.round(y_pred)

        # Ищем True Positives, False Positives и False Negatives
        tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

        # Вычисляем precision и recall
        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())

        # Вычисляем F1 score
        f1_score = 2 * precision * recall / (precision + recall + K.epsilon())

        # Усредняем F1 score по классам и возвращаем результат
        return K.mean(f1_score)

    def load_model_with_focal_loss(model_path):
        custom_objects = {'focal_loss': focal_loss,
                          'f1_score': f1_score}  # Добавляем пользовательскую функцию потерь в custom_objects
        model = tf.keras.models.load_model(model_path, compile=False)
        return model

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
    output_layer = Dense(3, activation='softmax')(dropout)

    model = Model(inputs=[branch1.input, branch2.input], outputs=output_layer)

    def predict(text, model_path, token_path):

        model = load_model_with_focal_loss(model_path)

        with open(token_path, 'rb') as f:
            tokenizer = pickle.load(f)

        sequences = tokenizer.texts_to_sequences(text)
        x_new = pad_sequences(sequences, maxlen=512)
        predictions = model.predict([x_new, x_new])

        # Использование numpy.argmax для определения предсказанного класса
        predicted_labels = np.argmax(predictions, axis=1)

        return predicted_labels

    return predict(str_array, '/home/egore/proj/cc/nlp_obj.h5', '/home/egore/proj/cc/tokenizer_obj.pkl')


text = ['Ужасный преподаватель', 'Хороший вебинар', 'Продуманная программа курса']
class_obj = predict_obj(text)
print(class_obj)
