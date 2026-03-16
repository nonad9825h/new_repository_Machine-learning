import json

import cv2
import numpy as np
import streamlit as st
import requests
import os
import pandas as pd
from PIL import Image
import plotly.express as px
import io
from streamlit_drawable_canvas import st_canvas

# Обнуление статусов прокси
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

# Настройка страницы
st.set_page_config(
    page_title="Определение токсичности вашего комментария",
    initial_sidebar_state="expanded"
)


# Основной интерфейс
def main():
    st.title('Проект с комментариями')
    st.subheader('Распознавания токсичности комментария')

    input_text_cm = st.text_input(
        'Введите ваш комментарий:',
        placeholder="Напишите ваш текст тут...", )

    if st.button('Определить токсичность', type="primary"):
        if input_text_cm == '':
            st.warning('💬 Пожалуйста, укажите ваш комментарий!!!')
        else:
            with st.spinner('Анализ комментария...'):
                url = 'http://127.0.0.1:8000/predict'
                data = {
                    'text': str(input_text_cm)
                }
                print(data)

                try:
                    response = requests.post(url, json=data)
                    result = response.json()

                    qwe123 = result.get('sentiment')
                    ewq321 = result.get('confidence')

                    # st.write(qwe123)
                    # st.write(ewq321)
                    #
                    # print(qwe123)
                    # print(ewq321)

                    ewq321 = float(ewq321.replace('[', '').replace(']', ''))
                    # st.write(ewq321)

                    sentiment = "Токсичный" if ewq321 < 0.5 else "Не токсичный"
                    confidence = ewq321 if ewq321 > 0.5 else 1 - ewq321

                    # st.write(sentiment)
                    # st.write(confidence)

                    st.header('Определенный класс:')
                    st.metric(label=f'{qwe123}', value=f'{confidence}%')

                    if sentiment == 'Токсичный':
                        conf1 = confidence
                        conf2 = 1 - confidence
                    else:
                        conf1 = 1 - confidence
                        conf2 = confidence

                    data_conf = {
                        'Тип': ['Токсичный', 'Не токсичный'],
                        'Вероятности': [conf1, conf2],
                    }
                    df_conf = pd.DataFrame(data_conf)

                    # диаграмма
                    fig = px.pie(
                        df_conf,
                        values='Вероятности',
                        names='Тип',
                        color_discrete_sequence=px.colors.sequential.Viridis
                    )
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        hovertemplate='<b>%{label} %{percent}</b>'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.write('Распределение вероятностей:')
                    st.data_editor(df_conf)

                    if 'history' not in st.session_state:
                        st.session_state.history = []

                    st.session_state.history.append({
                        'Комментарий': input_text_cm,
                        'Токсичность': qwe123,
                        'Вероятность': round(confidence, 8)
                    })

                except requests.exceptions.RequestException as e:
                    st.error(f'❌ Ошибка подключения к API: {e}')

    # Показ истории запросов
    if 'history' in st.session_state and st.session_state.history:
        st.subheader("История запросов:")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(
            history_df,
            column_config={
                'topic': 'Тема',
                'confidence': 'Вероятность',
                'text': 'Текст'
            },
            hide_index=False,
            use_container_width=True
        )


if __name__ == "__main__":
    main()
