from nltk import word_tokenize
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
import re


def freq_word(sentence: str):

    def remove_chars_from_text(text, chars):
        return "".join([ch for ch in text if ch not in chars])

    def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', text)

    nltk.download('punkt')
    nltk.download('stopwords')
    sentence = remove_punctuation(sentence)
    spec_chars = string.punctuation + '\n\xa0«»\t—…'
    sentence = "".join([ch for ch in sentence if ch not in spec_chars])
    sentence = remove_chars_from_text(sentence, spec_chars)
    sentence = remove_chars_from_text(sentence, string.digits)
    text_tokens = word_tokenize(sentence)
    sentence = nltk.Text(text_tokens)
    russian_stopwords = stopwords.words("russian")
    russian_stopwords.extend(['это', 'нею', 'в'])
    fdist = FreqDist(sentence)
    text_raw = " ".join(sentence)
    wordcloud = WordCloud(stopwords=set(russian_stopwords)).generate(text_raw)
    return wordcloud


cloud = freq_word('Привет красивый мир, мир очень красив')
plt.figure(figsize=(6, 6), dpi=2000) 
plt.imshow(cloud)
plt.axis("off")
plt.show()