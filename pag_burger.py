import streamlit as st
import plotly.figure_factory as ff
from PIL import Image
import pandas as pd
import numpy as np
import joblib
from prophet.plot import plot_plotly
from datetime import datetime
import matplotlib.pyplot as plt

def main():
    st.button("Re-run")
    # set up layout
    st.title("Benvenuto nella sezione dei burger!")
    st.markdown("Di seguito alcune informazioni sulle vendite dei burger")

    
    data=pd.read_excel('dati.xlsx', 
                     index_col = 'Calendario',
                     parse_dates = True
                     )
    df_burger = data[data['Tipologia'] == 'Burger']
    df_burger = df_burger.drop('Tipologia', axis = 1)
    df_burger = df_burger[(df_burger.index >= '2021-05-25') & (df_burger.index <= '2022-11-05')]

    st.markdown('Distribuzione dei dati usati')
    fig, ax = plt.subplots()
    ax.hist(df_burger['Quantita'], bins=100)
    plt.xlabel('Quantità vendute di burger')
    plt.ylabel('Frequenza')
    st.pyplot(fig)

    df_burger = df_burger.reset_index()
    df_burger.columns = ['ds', 'y']



    model = joblib.load('model_burger.pkl')


    st.subheader('Previsioni')
    add_slider = st.slider('Scegli il periodo di previsione dei dati', 10, 150)
    future = model.make_future_dataframe(int(add_slider), freq='D')
    forecast = model.predict(future)


    fig = plot_plotly(model, forecast)
    fig.update_layout(yaxis_title = 'Quantità di burger venduti',
                xaxis_title = 'Data',
                title = 'Vendita Burger'
                )
    st.plotly_chart(fig)



    if st.button('Mostrare componenti dei trend'):
        st.pyplot(model.plot_components(forecast))
    else:
        print('Niente')




if __name__ == "__main__":
    main()


