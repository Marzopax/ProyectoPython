import streamlit as st
import pandas as pd
from transformers import pipeline

# 1. Configuración de la IA (Clase 6 - LLMs & Hugging Face)
# Usamos un modelo liviano en español para análisis de sentimientos
@st.cache_resource
def iniciar_pipeline():
    return pipeline("sentiment-analysis", model="pysentimiento/robertuito-sentiment-analysis")

classifier = iniciar_pipeline()

st.title("🎫 Sistema Inteligente de Soporte")
st.write("Subí un CSV con tickets, el sistema los limpia y clasifica.")

# Requisito: Carga de datos interactiva
uploaded_file = st.file_uploader("Subir archivo de tickets (CSV)", type=["csv"])

if uploaded_file is not None:
    # 2. PROCESAMIENTO CON PANDAS (Clase 2 - Limpieza)
    df = pd.read_csv(uploaded_file)
    
    # Técnicas de limpieza de la Clase 2
    df.dropna(subset=['descripcion'], inplace=True) # Elimina nulos
    df.drop_duplicates(inplace=True)               # Elimina duplicados
    df['descripcion'] = df['descripcion'].str.strip()
    
    st.write("### Datos Limpios con Pandas", df.head())
    
    if st.button("🚀 Ejecutar Análisis IA"):
        with st.spinner("Procesando..."):
            # 3. INTEGRACIÓN CON LLM (Clase 6)
            # Inferencia aplicada al DataFrame procesado
            def clasificar(texto):
                res = classifier(texto)[0]
                # Traducimos etiquetas técnicas a algo útil
                label = res['label']
                return "Positivo" if label == "POS" else ("Negativo" if label == "NEG" else "Neutral")

            df['sentimiento'] = df['descripcion'].apply(clasificar)
            
            st.success("¡Análisis completado!")
            st.dataframe(df)
            
            # Descarga del resultado
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar CSV Clasificado", csv, "tickets_ia.csv", "text/csv")