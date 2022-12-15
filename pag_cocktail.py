import streamlit as st
import pandas as pd
from prophet.plot import plot_plotly
import joblib
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import acf


def main():
    st.title('Dataframe originale sui cocktails')
    df = pd.read_excel('dati.xlsx', parse_dates=True)
    copia_df=df.copy()
    df_cocktail=copia_df[copia_df['Tipologia']=='Cocktail']
    df_cocktail=df_cocktail.drop('Tipologia',axis=1)
    df_cocktail['Data'] = df_cocktail['Calendario']
    df_cocktail = df_cocktail.drop('Calendario',axis=1)
    quanti = st.slider('Scegli quanti dati vedere',1,len(df_cocktail),1)
    df_cocktail = df_cocktail[['Data','Quantita']]
    #df_cocktail['Quantita']=round(df_cocktail['Quantita'])
    df_cocktail['Data'] = pd.to_datetime(df_cocktail['Data']).dt.date
    df_cocktail['Data']=pd.to_datetime(df_cocktail['Data'],format='%Y/%m/%d')
    df_cocktail['Data']=df_cocktail['Data'].dt.strftime('%d/%m/%Y')
    df_cocktail.columns=['Data','Quantità']
    st.dataframe(df_cocktail.head(quanti))

    st.title('Descrizione dati:')
    st.dataframe(df_cocktail.describe())

    st.title('Istogramma originale sui cocktails')
    bins_scelti = st.slider('Selezionare i bins',
                            1, 
                            len(df_cocktail), 
                            1)
    isto = df_cocktail['Quantità'].plot(kind='hist', bins=bins_scelti)
    st.pyplot(isto.figure,clear_figure=True)

    st.title('Lag plot originale sui cocktails')
    lag = int(st.text_input(f'Scegli il lag (dev\'essere un numero intero compreso tra 1 e {len(df_cocktail)-1}!)',value=1))
    if (lag>len(df_cocktail)-1) or (lag < 1):
        st.write(f'Devi inserire un numero compreso tra 1 e {len(df_cocktail)-1}!')
    else:
        grafico_lag = lag_plot(df_cocktail['Quantità'],lag)
        st.pyplot(grafico_lag.figure,clear_figure=True)

        autocorrelation_vet = acf(df_cocktail['Quantità'],nlags=lag)
        autocorrelation = autocorrelation_vet[-1]
        st.write(f'L\'autocorrelazione di questo lag plot è del {round(100*autocorrelation,2)}%')
    

    st.title('Dataframe sui cocktails senza outliers')
    # Calcola la media e la deviazione standard
    mean = df_cocktail['Quantità'].mean()
    std = df_cocktail['Quantità'].std()

    #Q1 = df_cocktail['Quantità'].quantile(0.25)
    #Q3 = df_cocktail['Quantità'].quantile(0.75)

    # Crea l'intervallo di deviazione standard
    lower = mean - std   
    upper = mean + 2 * std  
    #lower = Q1-1.5*(Q3-Q1)
    #upper = Q3+1.5*(Q3-Q1)

    df_cocktail_rid = df_cocktail[(df_cocktail['Quantità']>lower) & (df_cocktail['Quantità']<upper)]
    df_cocktail_rid = df_cocktail_rid.reset_index()
    df_cocktail_rid = df_cocktail_rid.drop('index',axis=1)
    quanti2 = st.slider('Scegli quanti dati vedere',1,len(df_cocktail_rid),1)
    st.dataframe(df_cocktail_rid.head(quanti2))

    st.title('Descrizione dati:')
    st.dataframe(df_cocktail_rid.describe())

    st.title('Istogramma sui cocktails senza outliers')
    bins_scelti2 = st.slider('Selezionare i bins',
                            1, 
                            len(df_cocktail_rid),
                            1)
    isto2 = df_cocktail_rid['Quantità'].plot(kind='hist', bins=bins_scelti2)
    st.pyplot(isto2.figure,clear_figure=True)

    st.title('Lag plot sui cocktails senza outliers')
    lag2 = int(st.text_input(f'Scegli il lag (dev\'essere un numero intero compreso tra 1 e {len(df_cocktail_rid)-1}!)',value=1))
    if (lag2>len(df_cocktail_rid)-1) or (lag2 < 1):
        st.write(f'Devi inserire un numero compreso tra 1 e {len(df_cocktail_rid)-1}!')
    else:
        grafico_lag2 = lag_plot(df_cocktail_rid['Quantità'],lag2)
        st.pyplot(grafico_lag2.figure,clear_figure=True)
        
        autocorrelation_vet2 = acf(df_cocktail_rid['Quantità'],nlags=lag2)
        autocorrelation2 = autocorrelation_vet2[-1]
        st.write(f'L\'autocorrelazione di questo lag plot è del {round(100*autocorrelation2,2)}%')

    model = joblib.load('model_cocktail.pkl')

    st.title('Componenti dei cocktail senza outliers')
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
    fig.update_layout(title="Previsione dei cocktails venduti",
                    yaxis_title='cocktails venduti',
                    xaxis_title="Data",
                    )
    st.plotly_chart(fig)

    # df_cv_final=cross_validation(model,
    #                         horizon="60 days",
    #                         period='10 days',
    #                         initial='450 days',
    #                         )
    # df_performance=performance_metrics(df_cv_final)
    # mape = df_performance['mape'].mean()
    # st.write(f'L\'errore percentuale medio della previsione calcolato sugli ultimi 60 giorni è del {round(mape*100,2)}%')
    st.write(f'L\'errore percentuale medio della previsione calcolato sugli ultimi 60 giorni è del 20,7%')

if __name__ == "__main__":
    main()
