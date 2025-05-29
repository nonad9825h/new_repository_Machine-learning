from fastapi import FastAPI, HTTPException
import pickle
from pydantic import BaseModel
import string
import re
import nltk
# nltk.download('stopwords')
# nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3
import traceback

app = FastAPI()
morph = pymorphy3.MorphAnalyzer(lang='ru')

# Загрузка модели и векторайзера
try:
    # # изначальная модель
    # with open('model/art/model_lr_art_v1.pkl', 'rb') as file:
    #     model = pickle.load(file)
    # with open('model/art/tfidf_model_vectorizer_art_v1.pkl', 'rb') as file:
    #     vectorizer = pickle.load(file)
    # print("Модель и векторайзер успешно загружены")

    # сбалансированные данные
    with open('model/art/model_knn_art_v2mn.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('model/art/tfidf_model_vectorizer_art_v2mn.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    print("Модель и векторайзер успешно загружены")

except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    raise RuntimeError("Не удалось загрузить модель") from e

# Загрузка стоп-слов один раз при старте приложения
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

russian_stopwords = stopwords.words("russian")
russian_stopwords.extend(['т.д.', 'т', 'д', 'это', 'который', 'с', 'своём', 'всем',
                         'наш', 'свой', 'такой', 'другой', 'поэтому', 'также',
                         'например', 'мочь', 'почему', 'которой', 'которому'])


def fun_punctuation_text(text: str) -> str:
    """
    Методы предобработки текста
    """
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    text = ''.join([i if not i.isdigit() else '' for i in text])
    text = ''.join([i if i.isalpha() else ' ' for i in text])
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub('[a-z]', '', text, flags=re.I)  # Удаление латинских букв
    st = '}\xa0'
    text = ''.join([ch if ch not in st else ' ' for ch in text])
    return text.strip()


def fun_lemmatizing_text(text: str) -> str:
    """
    Лемматизация текста
    """
    tokens = word_tokenize(text)
    res = []
    for word in tokens:
        p = morph.parse(word)[0]
        res.append(p.normal_form)
    return ' '.join(res)


def fun_tokenize(text: str) -> str:
    """
    Токенизация и удаление стоп-слов
    """
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in russian_stopwords]
    return ' '.join(tokens)


def fun_pred_text(text: str) -> str:
    """
    Комплексная предобработка текста
    """
    text = fun_punctuation_text(text)
    text = fun_lemmatizing_text(text)
    text = fun_tokenize(text)
    return text


def predict_cluster(text: str) -> tuple:
    """
    Функция предсказания кластера
    """
    try:
        processed_text = fun_pred_text(text)
        text_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]

        mapping = {
            0: 'Человек и работа',
            1: 'Дизайн',
            2: 'Приложение и работа',
            3: 'Московский проект премии рунета',
            4: 'Сервис компании водителей яндекс такси',
            5: 'Работа человека в компании',
            6: 'Машина/робот и человек'
        }

        selected_cluster = mapping[prediction]
        confidence_cluster = f'{probabilities[prediction]:.7}'

        confidence = (f'Работа - {probabilities[0]:.7}'
                      f'\n\nДизайн - {probabilities[1]:.7}\n'
                      f'\n\nПриложение и работа - {probabilities[2]:.7}'
                      f'\n\nПремии, номинации - {probabilities[3]:.7}'
                      f'\n\nКомпания-сервис - {probabilities[4]:.7}'
                      f'\n\nРабота человека в компании - {probabilities[5]:.7}'
                      f'\n\nМашина/робот и человек - {probabilities[6]:.7}'
                      )
        print(probabilities)
        return [selected_cluster, confidence_cluster], confidence
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")


# Модель для входных данных
class Item(BaseModel):
    text: str


@app.post('/predict')
async def post_pred_text(item: Item):
    try:
        cluster, confidence = predict_cluster(item.text)
        return {
            'cluster': cluster,
            'confidence': confidence
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Добавляем корневой эндпоинт
@app.get("/")
def read_root():
    return {"message": "Добро пожаловать в API. Swagger доступен по /docs"}
