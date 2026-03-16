import ast
import asyncio
import json
import os
import numpy as np
import pandas as pd
import aiomysql
from fastapi import FastAPI, HTTPException, UploadFile, File
import pickle
import string
import traceback
from keras.models import load_model
import numpy as np
from keras.preprocessing import text, sequence
import re
import nltk
import pymorphy3
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import mysql.connector
from mysql.connector.aio import connect

app = FastAPI()

stemmer = SnowballStemmer("russian")
morph = pymorphy3.MorphAnalyzer(lang='ru')

np.set_printoptions(suppress=True)
try:
    model_l = load_model("rcc_nlp_models\\models\\model_v3e_nlp_comment.h5", compile=False)
    with open("rcc_nlp_models\\models\\keras_token_v3e_rnn_nlp.pkl", 'rb') as file:
        token_rnn = pickle.load(file)
    print('===== Модель и векторайзер успешно загружены!! =====')
except Exception as e:
    print(f"Ошибка загрузки данных: {e}")
    raise RuntimeError("Не удалось загрузить данные") from e

string.number = '1234567890'
string.punctuation = "#$%&\'()*+,-./:;<=>!?@[\\]^_`{|}~«»—"
russian_stopwords = stopwords.words("russian")
russian_stopwords.extend(text for text in ['т.д.', 'т', 'д', 'это', 'который', 'с', 'своём', 'всем', 'наш', 'свой',
                                           'такой', 'другой', 'поэтому', 'также', 'например', 'мочь', 'почему',
                                           'которой', 'которому', 'которому',
                                           'человек', 'просто', 'ещё', 'народ', 'год', 'очень'])

# @app.on_event("start_up")
# async def database_connect():
#     try:
#         config_conn = await connect(
#             host="localhost",
#             user="root",
#             password="admin",
#             database="py_rcc_nlp_comment"
#         )
#         db = await config_conn.cursor()
#         print("Соединение с базой данных установлено!!")
#         # await config_conn.close()
#         # await db.close()
#     except mysql.connector.Error as err:
#         print(f"Ошибка: {err}")

config = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "admin",
    "db": "py_rcc_nlp_comment",
    "autocommit": True
}

app.db_conn = None


@app.on_event("startup")
async def startup():
    # app.db_conn = await aiomysql.create_pool(host='localhost', port='3306', user='root', password='admin',
    #                                          db='py_rcc_nlp_comment', autocommit=True)
    app.db_conn = await  aiomysql.create_pool(**config)
    print("Соединение с базой данных установлено!!")


@app.on_event("shutdown")
async def shutdown():
    app.db_conn.close()
    await app.db_conn.wait_closed()


@app.get("/test_api")
async def test_api():
    async with app.db_conn.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT * FROM py_rcc_nlp_comment.nlp_comment")
            result = await cursor.fetchall()
            print(result)
            return {"comment": result}


def remove_number(text):
    """ Функция удаления цифр """
    try:
        return "".join([char for char in text if char not in string.number])
    except:
        return text


def remove_sing_char(text):
    """ Функция одиночных символьных комментариев """
    try:
        if len(text) == 1:
            return None
        return text
    except:
        return text


def text_toLowerCase(text: str):
    """ Функция приведения к нижнему регистру """
    try:
        return text.lower()
    except:
        return text


def remove_punct(text):
    """ Функция удаления пунктуации """
    try:
        return "".join([char for char in text if char not in string.punctuation])
    except:
        return text


def remove_latin_symbols(text):
    """ Функция латинских символов """
    return re.sub('[a-z]', '', str(text), flags=re.I)


def remove_stopwords_russian(text):
    """ Функция удаления стоп-слов """
    try:
        t = word_tokenize(text)
        tokens = [token for token in t if token not in russian_stopwords]
        text = " ".join(tokens)
        return text
    except:
        return text


def stemmer_text(text):
    """ Функция стемминга """
    try:
        t = word_tokenize(text)
        tokens = [stemmer.stem(token) for token in t if token not in russian_stopwords]
        text = " ".join(tokens)
        return text
    except Exception as e:
        return text


def lemmatizing_text(text):
    """ Функция лемматизации """
    try:
        t = word_tokenize(text)
        result = list()
        for word in t:
            p = morph.parse(word)[0]
            result.append(p.normal_form)
        text = " ".join(result)
        return text
    except:
        return text


async def nlp_input_system(comment):
    """ Функция определения токсичности комментария """
    try:
        max_features = 40000
        max_length = 300

        # Обработка текста
        text_comment = [remove_number(text) for text in comment]
        text_comment = [text_toLowerCase(text) for text in text_comment]
        text_comment = [remove_punct(text) for text in text_comment]
        text_comment = [remove_latin_symbols(text) for text in text_comment]
        text_comment = [remove_stopwords_russian(text) for text in text_comment]
        text_comment = [lemmatizing_text(text) for text in text_comment]

        # повторное удаление одиночных символов и стоп-слов после лемматизации
        text_comment = [remove_sing_char(text) for text in text_comment]
        text_comment = [remove_stopwords_russian(text) for text in text_comment]

        print(comment)
        print(f'{text_comment}\n')

        test_text = token_rnn.texts_to_sequences(text_comment)
        sq_test_text = sequence.pad_sequences(test_text, maxlen=max_length)
        prediction = model_l.predict(sq_test_text, verbose=1, batch_size=2)

        print(f"{prediction}")
        sentiment = "Токсичный" if prediction < 0.5 else "Не токсичный"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        print(f"Тональность: {sentiment}\nВероятность: {confidence[0]}")

        # # отправка комментария в базу данных
        # conn = mysql.connector.connect(
        #     host="localhost",
        #     user="root",
        #     password="admin",
        #     database="py_rcc_nlp_comment"
        # )
        # conn.close()
        # print("Соединение с базой данных установлено!!")

        # request = f"INSERT INTO `py_rcc_nlp_comment`.`nlp_comment` (`comment`, `toxic_type`, `confidence`) VALUES (%s, %s, %s);"
        # content = ("иди в жопу тварь", "Не токсичный", 0.9877777)
        # db.execute(request, content)
        # confidence.commit()

        await cmt_add(comment, sentiment, confidence[0])

        return sentiment, prediction[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")


class Item(BaseModel):
    text: str


class CommentDb():
    comment: str
    toxic_type: str
    confidence: float


@app.post("/cmt_add")
async def cmt_add(comment, toxic_type, confidence):
    async with app.db_conn.acquire() as conn:
        async with conn.cursor() as cursor:
            # print(comment.comment)
            print(f'\nотправка данных в бд')
            print(comment[0])
            print(toxic_type)
            print(confidence[0])

            request = f"INSERT INTO `py_rcc_nlp_comment`.`nlp_comment` (`comment`, `toxic_type`, `confidence`) VALUES (%s, %s, %s);"
            content = (comment[0], toxic_type, confidence[0])
            await cursor.execute(request, content)
            return {"comment_add": content}


@app.post('/predict')
async def post_pred_text(item: Item):
    try:
        text = item.text
        print('eqweqweqw')
        print(text)

        sentiment, confidence = await nlp_input_system([text])
        # print(sentiment)
        # print(confidence)

        return {
            'sentiment': str(sentiment),
            'confidence': str(confidence)
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Добавляем корневой эндпоинт
@app.get("/")
def read_root():
    return {"message": "Добро пожаловать в API. Swagger доступен по /docs"}
