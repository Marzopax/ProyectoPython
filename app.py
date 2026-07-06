import streamlit as st
import pandas as pd
from transformers import pipeline

@st.cache_resource
def iniciar_pipeline():
    return pipeline("sentiment-analysis", model="pysentimiento/robertuito-sentiment-analysis")

classifier = iniciar_pipeline()

st.title("🎫 Sistema Inteligente de Soporte")

uploaded_file = st.file_uploader("Subir archivo de tickets (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # NUEVA VALIDACIÓN: Mostrar columnas disponibles para el usuario
    st.write("Columnas detectadas en el archivo:", list(df.columns))
    
    # Cambiamos 'descripcion' por el nombre real de tu columna (ajustar si es necesario)
    target_col = 'descripcion' 
    
    if target_col in df.columns:
        # 2. PROCESAMIENTO CON PANDAS (Clase 2)
        df.dropna(subset=[target_col], inplace=True) 
        df.drop_duplicates(inplace=True)               
        df[target_col] = df[target_col].astype(str).str.strip()
        
        st.write("### Datos Limpios", df.head())
        
        if st.button("🚀 Ejecutar Análisis IA"):
            with st.spinner("Procesando..."):
                def clasificar(texto):
                    res = classifier(texto)[0]
                    return "Positivo" if res['label'] == "POS" else ("Negativo" if res['label'] == "NEG" else "Neutral")

                df['sentimiento'] = df[target_col].apply(clasificar)
                st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Descargar CSV Clasificado", csv, "tickets_ia.csv", "text/csv")
    else:
        st.error(f"Error: El archivo debe contener una columna llamada '{target_col}'. Verifica tu archivo CSV.")