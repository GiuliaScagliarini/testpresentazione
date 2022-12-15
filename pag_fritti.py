import streamlit as st
import pandas as pd
from prophet.plot import plot_plotly
import joblib
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import acf

def main():
    st.title('Dataframe originale sui fritti')
    df = pd.read_excel('dati.xlsx', parse_dates=True)
    copia_df=df.copy()
    df_fritti=copia_df[copia_df['Tipologia']=='Fritti']
    df_fritti=df_fritti.drop('Tipologia',axis=1)
    df_fritti['Data'] = df_fritti['Calendario']
    df_fritti = df_fritti.drop('Calendario',axis=1)
    quante = st.slider('Scegli quanti dati vedere',1,len(df_fritti),1)
    df_fritti = df_fritti[['Data','Quantita']]
    #df_fritti['Quantita']=round(df_fritti['Quantita'])
    df_fritti['Data'] = pd.to_datetime(df_fritti['Data']).dt.date
    df_fritti['Data']=pd.to_datetime(df_fritti['Data'],format='%Y/%m/%d')
    df_fritti['Data']=df_fritti['Data'].dt.strftime('%d/%m/%Y')
    df_fritti.columns=['Data','Quantità']
    df_fritti = df_fritti.reset_index()
    df_fritti = df_fritti.drop('index',axis=1)
    st.dataframe(df_fritti.head(quante))

    st.title('Riepilogo')
    st.dataframe(df_fritti.describe())

    st.title('Istogramma originale sui fritti')
    bins_scelti = st.slider(
        'Selezionare i bins',
        1, len(df_fritti), 1)
    isto = df_fritti['Quantità'].plot(kind='hist', bins=bins_scelti)
    st.pyplot(isto.figure,clear_figure=True)

    st.title('Lag plot originale sui fritti')
    lag = int(st.text_input(f'Scegli il lag (dev\'essere un numero intero compreso tra 1 e {len(df_fritti)-1}!)',value=1))
    if (lag>len(df_fritti)-1) or (lag < 1):
        st.write(f'Devi inserire un numero compreso tra 1 e {len(df_fritti)-1}!')
    else:
        grafico_lag = lag_plot(df_fritti['Quantità'],lag)
        st.pyplot(grafico_lag.figure,clear_figure=True)

        autocorrelation_vet = acf(df_fritti['Quantità'],nlags=lag)
        autocorrelation = autocorrelation_vet[-1]
        st.write(f'L\'autocorrelazione di questo lag plot è del {round(100*autocorrelation,2)}%')
    

    st.title('Dataframe sui fritti senza outliers')
    df_fritti_rid = df_fritti[(df_fritti['Quantità']<69) & (df_fritti['Quantità']>5)]
    df_fritti_rid = df_fritti_rid.reset_index()
    df_fritti_rid = df_fritti_rid.drop('index',axis=1)
    quante2 = st.slider('Scegli quanti dati vedere',1,len(df_fritti_rid),1)
    st.dataframe(df_fritti_rid.head(quante2))

    st.title('Riepilogo')
    st.dataframe(df_fritti_rid.describe())

    st.title('Istogramma sui fritti senza outliers')
    bins_scelti2 = st.slider(
        'Selezionare i bins',
        1, len(df_fritti_rid), 1)
    isto2 = df_fritti_rid['Quantità'].plot(kind='hist', bins=bins_scelti2)
    st.pyplot(isto2.figure,clear_figure=True)

    st.title('Lag plot sui fritti senza outliers')
    lag2 = int(st.text_input(f'Scegli il lag (dev\'essere un numero intero compreso tra 1 e {len(df_fritti_rid)-1}!)',value=1))
    if (lag2>len(df_fritti_rid)-1) or (lag2 < 1):
        st.write(f'Devi inserire un numero compreso tra 1 e {len(df_fritti_rid)-1}!')
    else:
        grafico_lag2 = lag_plot(df_fritti_rid['Quantità'],lag2)
        st.pyplot(grafico_lag2.figure,clear_figure=True)
        
        autocorrelation_vet2 = acf(df_fritti_rid['Quantità'],nlags=lag2)
        autocorrelation2 = autocorrelation_vet2[-1]
        st.write(f'L\'autocorrelazione di questo lag plot è del {round(100*autocorrelation2,2)}%')

    model = joblib.load('model_fritti.pkl')

    st.title('Componenti dei fritti senza outliers')
    quanto_trend = st.slider('Scegli per quanti giorni nel futuro vuoi vedere il trend',0,365,1)
    future = model.make_future_dataframe(periods=quanto_trend)
    forecast = model.predict(future)
    comp = model.plot_components(forecast)
    st.pyplot(comp.figure,clear_figure=True)

    st.title('Forecasting')
    da_pred = st.slider('Scegli quanti giorni prevedere',1,365,1)
    future = model.make_future_dataframe(da_pred, freq='D')
    forecast = model.predict(future)
    fig = plot_plotly(model, forecast)
    fig.update_layout(title="Previsione dei fritti venduti",
                    yaxis_title='Fritti venduti',
                    xaxis_title="Data",
                    )
    st.plotly_chart(fig)

    # df_cv_final=cross_validation(model,
    #                         horizon=str(da_pred) + " days",
    #                         period='10 days',
    #                         initial='450 days',
    #                         )
    # df_performance=performance_metrics(df_cv_final)
    # mape = df_performance['mape'].mean()
    # st.write(f'L\'errore percentuale medio è del {round(mape*100,2)}%')
    st.write(f'L\'errore percentuale medio della previsione calcolato sugli ultimi 60 giorni è intorno al 36%')


if __name__ == "__main__":
    main()
