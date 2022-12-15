import streamlit as st
import pandas as pd
from prophet.plot import plot_plotly
import joblib
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import acf
from prophet.diagnostics import cross_validation, performance_metrics
from datetime import date

def main():
    pd.options.display.float_format = '{:,.1f}'.format
    st.subheader('Tabella dati input - Aggregazione giornaliera')
    df = pd.read_excel('dati.xlsx', parse_dates=True,dtype={"Quantita":int})
    copia_df=df.copy()
    df_spine=copia_df[copia_df['Tipologia']=='Spine']
    df_spine=df_spine.drop('Tipologia',axis=1)
    df_spine['Data'] = df_spine['Calendario']
    df_spine = df_spine.drop('Calendario',axis=1)
    quante = st.slider("Selezionare la dimensione del dataset",100,len(df_spine),250)
    df_spine = df_spine[['Data','Quantita']]
    df_spine['Data'] = pd.to_datetime(df_spine['Data']).dt.date
    df_spine['Data']=pd.to_datetime(df_spine['Data'],format='%Y/%m/%d')
    df_spine['Data']=df_spine['Data'].dt.strftime('%d/%m/%Y')
    df_spine.columns=['Data','Quantità']
    st.dataframe(df_spine.head(quante))

    st.subheader('Tabella statistica riassuntiva')
    st.dataframe(df_spine.describe().T)

    st.subheader('Istogramma sui dati di vendita delle spine')
    bins_scelti = st.slider(
        'Selezionare il numero di bins',
        40, 100, 60)
    isto = df_spine['Quantità'].plot(kind='hist', bins=bins_scelti)
    st.pyplot(isto.figure,clear_figure=True)

    st.subheader('Lag plot originale sulle spine')
    st.write ("""Un lag plot è un grafico utilizzato in statistica per individuare la presenza di autocorrelazione nei dati. L'autocorrelazione si riferisce alla dipendenza tra gli elementi di una serie temporale, ossia alla presenza di una relazione tra i valori di una variabile a distanza di un certo intervallo di tempo (detto lag). Se ad esempio utilizziamo un lag di 7 giorni, il primo punto avrà coordinate u = y(1) e v = y(1+7) = y(8), il secondo punto u = y(2) e v = y(2+7) = y(9) e così via, dove y(t) in questo caso è il numero di spine vendute al tempo t. Idealmente l'autocorrelazione è, in valore assoluto, uguale a 1, che è il caso in cui tutti i punti giacciono sulla stessa retta.\nTuttavia, è importante ricordare che il lag plot è tanto più inaffidabile quanto più è alto il numero di giorni mancanti nel dataset.""")
    lag = int(st.text_input(f'Scegli il lag (dev\'essere un numero intero compreso tra 1 e 365)',value=1))
    if (lag>365) or (lag<1):
        st.write(f'Devi inserire un numero intero compreso tra 1 e 365!')
    else:
        grafico_lag = lag_plot(df_spine['Quantità'],lag)
        st.pyplot(grafico_lag.figure,clear_figure=True)

        autocorrelation_vet = acf(df_spine['Quantità'],nlags=lag)
        autocorrelation = autocorrelation_vet[-1]
        st.write(f'L\'autocorrelazione di questo lag plot è del {round(100*autocorrelation,2)}%')    

    st.subheader('Dataset senza outliers')
    df_spine_rid = df_spine[(df_spine['Quantità']<370) & (df_spine['Quantità']>10)]
    df_spine_rid = df_spine_rid.reset_index()
    df_spine_rid = df_spine_rid.drop('index',axis=1)
    quante2 = st.slider("Selezionare la dimensione del dataset",100,len(df_spine_rid),250)
    st.dataframe(df_spine_rid.head(quante2))
    st.write("""Per evitare di considerare valori troppo grandi o troppo piccoli (outliers) che rischiano di fuorviare le previsioni, sono stati utilizzati solo i dati contenuti entro 2 deviazioni standard dalla media""")

    st.subheader('Tabella statistica riassuntiva senza outliers')
    st.dataframe(df_spine_rid.describe().T)

    st.subheader('Istogramma sulle spine senza outliers')
    bins_scelti2 = st.slider(
        'Selezionare il numero bins',
        40, 100, 60)
    isto2 = df_spine_rid['Quantità'].plot(kind='hist', bins=bins_scelti2)
    st.pyplot(isto2.figure,clear_figure=True)

    st.subheader('Lag plot sulle spine senza outliers')
    lag2 = int(st.text_input(f'Scegli il lag (dev\'essere un numero intero compreso tra 1 e 365)',value=7))
    if (lag2>365) or (lag2<1):
        st.write(f'Devi inserire un numero intero compreso tra 1 e 365!')
    else:
        grafico_lag2 = lag_plot(df_spine_rid['Quantità'],lag2)
        st.pyplot(grafico_lag2.figure,clear_figure=True)
        
        autocorrelation_vet2 = acf(df_spine_rid['Quantità'],nlags=lag2)
        autocorrelation2 = autocorrelation_vet2[-1]
        st.write(f'L\'autocorrelazione di questo lag plot è del {round(100*autocorrelation2,2)}%')

    model = joblib.load('model_spine.pkl')

    st.subheader('Componenti delle spine senza outliers')
    quanto_trend = st.slider('Scegliere il numero di giorni di cui vedere le componenti future',0,180,60)
    future = model.make_future_dataframe(periods=quanto_trend)
    forecast = model.predict(future)
    comp = model.plot_components(forecast)
    st.pyplot(comp.figure,clear_figure=True)

    st.subheader('Forecasting')
    da_pred = st.slider('Scegliere il numero di giorni da prevedere',0,180,60)
    future = model.make_future_dataframe(da_pred, freq='D')
    forecast = model.predict(future)
    fig = plot_plotly(model, forecast)
    fig.update_layout(title="Previsione delle spine vendute",
                    yaxis_title='Spine vendute',
                    xaxis_title="Data",
                    )
    fig.add_vline(x=date.today(), line_width=3, line_dash="dash", line_color="red")
    fig.update_layout(width=1200)
    st.plotly_chart(fig)

    df_cv_final=cross_validation(model,
                            horizon="60 days",
                            period='10 days',
                            initial='450 days',
                            )
    df_performance=performance_metrics(df_cv_final)
    mape = df_performance['mape'].mean()
    st.write(f'L\'errore percentuale medio della previsione calcolato sugli ultimi 60 giorni è del {round(mape*100,2)}%')

if __name__ == "__main__":
    main()
