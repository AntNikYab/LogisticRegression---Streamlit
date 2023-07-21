import streamlit as st
import pandas as pd
import numpy as np
from logreg import LogReg

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():
    st.title("Простой сервис для регрессии и визуализации данных")

    # Загрузка файла .csv
    st.sidebar.title("Загрузка данных")
    uploaded_file = st.sidebar.file_uploader("Загрузите файл .csv", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("Файл успешно загружен!")
        
        # Выбор фич и таргета
        st.sidebar.title("Выбор фич и таргета")
        features = st.sidebar.multiselect("Выберите фичи", data.columns.tolist())
        target = st.sidebar.selectbox("Выберите таргет (столбец, который нужно предсказать)", data.columns)
        

        # Проверка наличия выбранных фич и таргета
        if len(features) > 0 and target:
            X = data[features]
            y = data[target]
            ss = StandardScaler()
            X_train = pd.DataFrame(data=ss.fit_transform(X), columns=features)
            # Обучение регрессии
            model = LogReg()
            model.fit(X_train, y)

            # Отображение результатов регрессии (словаря весов)
            st.header("Результаты регрессии:")
            weights_dict = {feature: weight for feature, weight in zip(features, model.coef_)}
            weights_dict["intercept"] = model.intercept_
            st.write(weights_dict)

            # Визуализация данных
            st.header("Визуализация данных")

            plot_type = st.selectbox("Выберите тип графика", ["scatter plot", "bar plot", "line plot"])

            # Мульти-селект для выбора фич для графиков
            selected_features = st.multiselect("Выберите фичи для графиков", features)

            if len(selected_features) > 0:
                if plot_type == "scatter plot":
                    plot_scatter(X, y, selected_features)

                elif plot_type == "bar plot":
                    plot_bar(X, y, selected_features)

                elif plot_type == "line plot":
                    plot_line(X, y, selected_features)

def plot_scatter(X, y, selected_features):
    st.subheader("Scatter Plot")
    if len(selected_features) == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X[selected_features[0]], X[selected_features[1]], c=y, cmap="viridis")
        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1])
        plt.legend(handles=scatter.legend_elements()[0], labels=y.unique().tolist())
        st.pyplot(fig)        

def plot_bar(X, y, selected_features):
    st.subheader("Bar Plot")
    if len(selected_features) > 0:
        for feature in selected_features:
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.bar(X[feature], y)
            ax.set_xlabel(feature)
            ax.set_ylabel("Target")
            plt.tight_layout()
            st.pyplot(fig)
        

def plot_line(X, y, selected_features):
    st.subheader("Line Plot")
    if len(selected_features) > 0:
        for feature in selected_features:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(X[feature], y)
            ax.set_xlabel(feature)
            ax.set_ylabel("Target")
            plt.tight_layout()
            st.pyplot(fig)

if __name__ == "__main__":
    main()