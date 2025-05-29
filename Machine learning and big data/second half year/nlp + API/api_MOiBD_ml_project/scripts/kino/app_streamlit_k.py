import streamlit as st
import requests
import os
import pandas as pd
import matplotlib.pyplot as plt

# Обнуление статусов прокси
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

# Настройка страницы
st.set_page_config(
    page_title="Предсказание кластера фильма",
    initial_sidebar_state="expanded"
)


# Основной интерфейс
def main():
    st.title('Предсказание типа фильма IMDb')

    # Поле ввода
    input_text = st.text_area(
        'Введите описание фильма:',
        height=200,
        placeholder="Напишите ваш текст тут..."
    )

    if st.button('Определить тему', type="primary"):
        if not input_text.strip():
            st.warning('💬 Пожалуйста, введите описание!!!')
        else:
            with st.spinner('Анализ описания...'):
                data = {
                    'text': input_text
                }
                url = 'http://127.0.0.1:8000/predict'

                try:
                    response = requests.post(url, json=data)
                    result = response.json()

                    clust = result.get('cluster')
                    confidence = result.get('confidence')

                    st.subheader('Вероятности тем:')
                    st.write(f'Тема фильма: {clust[0]}')
                    st.write(f'{confidence}')

                    if 'history' not in st.session_state:
                        st.session_state.history = []

                    st.session_state.history.append({
                        'topic': clust[0],
                        'confidence': clust[1],
                        'text': input_text[:300] + ("..." if len(input_text) > 300 else "")
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
