import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

# Скачиваем нужные данные NLTK при первом запуске, если их ещё нет
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# PorterStemmer изначально разработан для английского языка.
# Для русских слов стемминг не будет работать корректно (слова останутся
# почти без изменений), но bag-of-words всё равно работает нормально —
# слова просто сравниваются "как есть" (после lower()).
# В будущем можно заменить на pymorphy2 для правильного русского стемминга.
stemmer = PorterStemmer()


def tokenize(sentence: str) -> list:
    """Разбить предложение на список токенов/слов в нижнем регистре."""
    return nltk.word_tokenize(sentence.lower())


def stem(word: str) -> str:
    """
    Привести слово к корневой форме (стемминг).
    Для русских слов фактически просто приводит к нижнему регистру.
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence: list, all_words: list) -> np.ndarray:
    """
    Построить вектор bag-of-words (мешок слов).

    Для каждого слова из словаря all_words ставим 1, если оно встречается
    в предложении, иначе 0. Получается числовой вектор фиксированной длины —
    именно его подаём на вход нейронной сети.
    """
    # Стеммируем слова входного предложения
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1.0
    return bag
