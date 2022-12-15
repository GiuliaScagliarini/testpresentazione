import streamlit as st
import pandas as pd
from prophet.plot import plot_plotly
import joblib
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import acf
from prophet.diagnostics import cross_validation,performance_metrics

def main():
    st.title('Dataframe originale sui bar')
    df = pd.read_excel('dati.xlsx', parse_dates=True)
    copia_df=df.copy()
    df_bar=copia_df[copia_df['Tipologia']=='Bar']
    df_bar=df_bar.drop('Tipologia',axis=1)
    df_bar['Data'] = df_bar['Calendario']
    df_bar = df_bar.drop('Calendario',axis=1)
    quante = st.slider('Scegli quanti dati vedere',1,len(df_bar),1)
    df_bar = df_bar[['Data','Quantita']]
    #df_bar['Quantita']=round(df_bar['Quantita'])
    df_bar['Data'] = pd.to_datetime(df_bar['Data']).dt.date
    df_bar['Data']=pd.to_datetime(df_bar['Data'],format='%Y/%m/%d')
    df_bar['Data']=df_bar['Data'].dt.strftime('%d/%m/%Y')
    df_bar.columns=['Data','Quantità']
    df_bar = df_bar.reset_index()
    df_bar = df_bar.drop('index',axis=1)
    st.dataframe(df_bar.head(quante))

    st.title('Riepilogo')
    st.dataframe(df_bar.describe())

    st.title('Istogramma originale sui bar')
    bins_scelti = st.slider(
        'Selezionare i bins',
        1, len(df_bar), 1)
    isto = df_bar['Quantità'].plot(kind='hist', bins=bins_scelti)
    st.pyplot(isto.figure,clear_figure=True)

    st.title('Lag plot originale sui bar')
    lag = int(st.text_input(f'Scegli il lag (dev\'essere un numero intero compreso tra 1 e {len(df_bar)-1}!)',value=1))
    if (lag>len(df_bar)-1) or (lag < 1):
        st.write(f'Devi inserire un numero compreso tra 1 e {len(df_bar)-1}!')
    else:
        grafico_lag = lag_plot(df_bar['Quantità'],lag)
        st.pyplot(grafico_lag.figure,clear_figure=True)

        autocorrelation_vet = acf(df_bar['Quantità'],nlags=lag)
        autocorrelation = autocorrelation_vet[-1]
        st.write(f'L\'autocorrelazione di questo lag plot è del {round(100*autocorrelation,2)}%')

    st.title('Dataframe sui bar senza outliers')
    df_bar_rid = df_bar[(df_bar['Quantità']<109)]
    df_bar_rid = df_bar_rid.reset_index()
    df_bar_rid = df_bar_rid.drop('index',axis=1)
    quante2 = st.slider('Scegli quanti dati vedere',1,len(df_bar_rid),1)
    st.dataframe(df_bar_rid.head(quante2))

    st.title('Riepilogo')
    st.dataframe(df_bar_rid.describe())

    st.title('Istogramma sui bar senza outliers')
    bins_scelti2 = st.slider(
        'Selezionare i bins',
        1, len(df_bar_rid), 1)
    isto2 = df_bar_rid['Quantità'].plot(kind='hist', bins=bins_scelti2)
    st.pyplot(isto2.figure,clear_figure=True)

    st.title('Lag plot sui bar senza outliers')
    lag2 = int(st.text_input(f'Scegli il lag (dev\'essere un numero intero compreso tra 1 e {len(df_bar_rid)-1}!)',value=1))
    if (lag2>len(df_bar_rid)-1) or (lag2 < 1):
        st.write(f'Devi inserire un numero compreso tra 1 e {len(df_bar_rid)-1}!')
    else:
        grafico_lag2 = lag_plot(df_bar_rid['Quantità'],lag2)
        st.pyplot(grafico_lag2.figure,clear_figure=True)
        
        autocorrelation_vet2 = acf(df_bar_rid['Quantità'],nlags=lag2)
        autocorrelation2 = autocorrelation_vet2[-1]
        st.write(f'L\'autocorrelazione di questo lag plot è del {round(100*autocorrelation2,2)}%')


    model = joblib.load('model_bar.pkl')

    st.title('Componenti dei bar senza outliers')
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
    fig.update_layout(title="Previsione dei bar venduti",
                    yaxis_title='Bar venduti',
                    xaxis_title="Data",
                    )
    st.plotly_chart(fig)

    df_cv_final=cross_validation(model,
                            horizon=str(da_pred) + " days",
                            period='10 days',
                            initial='450 days',
                            )
    df_performance=performance_metrics(df_cv_final)
    mape = df_performance['mape'].mean()
    st.write(f'L\'errore percentuale medio è del {round(mape*100,2)}%')
    #st.write(f'L\'errore percentuale medio della previsione calcolato sugli ultimi 60 giorni è intorno al 34%')


if __name__ == "__main__":
    main()
