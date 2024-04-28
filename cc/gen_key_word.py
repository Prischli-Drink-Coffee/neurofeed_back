import torch
from transformers import pipeline
import os
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast


def get_key_word(txt: str) -> str:
    # Зададим название выбронной модели из хаба
    MODEL_NAME = 'UrukHan/t5-russian-summarization'
    MAX_INPUT = 256

    # Загрузка модели и токенизатора
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def gen(txt):
        task_prefix = "Spell correct: "  # Токенизирование данных
        if type(txt) != list: txt = [txt]
        encoded = tokenizer(
            [task_prefix + sequence for sequence in txt],
            padding="longest",
            max_length=MAX_INPUT,
            truncation=True,
            return_tensors="pt",
        )

        predicts = model.generate(**encoded)  # # Прогнозирование

        txt = tokenizer.batch_decode(predicts, skip_special_tokens=True)  # Декодируем данные
        return txt

    return gen(txt)


text = 'Я построил большой красивый дом, не потому что люблю быть строителем, а потому что я полон духом'

txt = get_key_word(text)
print(txt)
