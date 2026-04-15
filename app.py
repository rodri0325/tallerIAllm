import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import tiktoken
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
from dotenv import load_dotenv

# Cargar variables de entorno (solo para local)
load_dotenv()

# Configuración de página
st.set_page_config(
    page_title="Taller LLM - Desmontando Transformers",
    page_icon="🧠",
    layout="wide"
)

# Título principal
st.title("🧠 Taller Técnico: Desmontando los LLMs")
st.markdown("### Deep Learning y Arquitecturas Transformer")
st.markdown("**Prof. Jorge Ivan Padilla Buritica - Universidad EAFIT**")
st.markdown("---")

# Inicializar cliente de Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    st.warning("⚠️ No se encontró API Key de Groq. Algunas funciones no estarán disponibles.")
    client = None

# ==================== MÓDULO 1: TOKENIZADOR ====================
st.header("📝 Módulo 1: El Laboratorio del Tokenizador")
st.markdown("Visualiza cómo el texto se convierte en tokens y sus IDs numéricos.")

col1, col2 = st.columns([2, 1])
with col1:
    texto_input = st.text_area("Ingresa tu texto:", 
                               "El rey, el hombre y la mujer fueron al castillo.",
                               height=100)

if texto_input:
    # Usar tokenizador de GPT-2 (similar a muchos LLMs)
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(texto_input)
    token_strings = [encoding.decode([t]) for t in tokens]
    
    with col2:
        st.metric("📏 Caracteres", len(texto_input))
        st.metric("🔢 Tokens", len(tokens))
        st.metric("📊 Ratio", f"{len(tokens)/len(texto_input):.2f} tokens/carácter")
    
    # Mostrar tokens coloreados
    st.markdown("### 🔤 Tokens detectados:")
    cols = st.columns(min(len(token_strings), 10))
    for i, token in enumerate(token_strings[:50]):
        with cols[i % len(cols)]:
            st.markdown(f"<span style='background-color:#e6f3ff;padding:5px;border-radius:5px;margin:2px;display:inline-block'><b>Token {i}:</b> '{token}'<br><code>ID: {tokens[i]}</code></span>", 
                       unsafe_allow_html=True)
    
    with st.expander("📋 Ver mapeo completo"):
        df_tokens = pd.DataFrame({
            "Posición": range(len(tokens)),
            "Token": token_strings,
            "Token ID": tokens
        })
        st.dataframe(df_tokens, use_container_width=True)

st.markdown("---")

# ==================== MÓDULO 2: EMBEDDINGS Y GEOMETRÍA ====================
st.header("📐 Módulo 2: Geometría de las Palabras (Embeddings)")
st.markdown("Visualización de embeddings en 2D usando PCA - Verificando: rey - hombre + mujer ≈ reina")

# Palabras por defecto
palabras_default = "rey, hombre, mujer, reina"
palabras_input = st.text_input("Ingresa palabras separadas por comas:", 
                               palabras_default)

if palabras_input:
    palabras = [p.strip() for p in palabras_input.split(",")]
    
    if st.button("🔮 Generar y visualizar embeddings", type="primary"):
        with st.spinner("Generando embeddings con simulación (para despliegue online)..."):
            # NOTA: Groq no provee embeddings directamente.
            # Para despliegue online, usamos embeddings simulados que respetan relaciones semánticas.
            # En producción real, usarías HuggingFace o OpenAI embeddings.
            
            # Simulación didáctica de embeddings en 3D
            np.random.seed(42)
            embedding_dim = 50
            
            # Crear relaciones semánticas simuladas
            embeddings_dict = {}
            for i, palabra in enumerate(palabras):
                # Base aleatoria
                vec = np.random.randn(embedding_dim)
                
                # Asignar relaciones especiales para rey, hombre, mujer, reina
                if palabra.lower() == "rey":
                    vec = np.array([5, 4, 3] + list(np.random.randn(embedding_dim-3)*0.5))
                elif palabra.lower() == "hombre":
                    vec = np.array([4, 2, 1] + list(np.random.randn(embedding_dim-3)*0.5))
                elif palabra.lower() == "mujer":
                    vec = np.array([3, 3, 2] + list(np.random.randn(embedding_dim-3)*0.5))
                elif palabra.lower() == "reina":
                    vec = np.array([5, 5, 4] + list(np.random.randn(embedding_dim-3)*0.5))
                else:
                    vec = np.random.randn(embedding_dim)
                
                embeddings_dict[palabra] = vec
            
            # Reducción a 2D con PCA
            vectors = np.array(list(embeddings_dict.values()))
            pca = PCA(n_components=2)
            vectors_2d = pca.fit_transform(vectors)
            
            # Crear DataFrame para Plotly
            df_emb = pd.DataFrame({
                'Palabra': palabras,
                'X': vectors_2d[:, 0],
                'Y': vectors_2d[:, 1]
            })
            
            # Verificar relación rey - hombre + mujer ≈ reina
            if all(p in embeddings_dict for p in ["rey", "hombre", "mujer", "reina"]):
                rey_vec = embeddings_dict["rey"]
                hombre_vec = embeddings_dict["hombre"]
                mujer_vec = embeddings_dict["mujer"]
                reina_vec = embeddings_dict["reina"]
                
                relacion = rey_vec - hombre_vec + mujer_vec
                similitud = cosine_similarity([relacion], [reina_vec])[0][0]
                
                st.info(f"🎯 **Relación semántica:** cos(rey - hombre + mujer, reina) = **{similitud:.3f}**")
                if similitud > 0.7:
                    st.success("✅ Se verifica la relación: rey - hombre + mujer ≈ reina")
                else:
                    st.warning("⚠️ La relación no se verifica claramente (posiblemente por embeddings simulados)")
            
            # Gráfico interactivo
            fig = px.scatter(df_emb, x='X', y='Y', text='Palabra', 
                            title="Embeddings en 2D (PCA)",
                            labels={'X': 'Componente Principal 1', 'Y': 'Componente Principal 2'},
                            size=[20]*len(df_emb), size_max=30)
            fig.update_traces(textposition='top center', marker=dict(size=15))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("💡 **Nota didáctica:** Para producción real, se usarían embeddings de HuggingFace (ej. 'all-MiniLM-L6-v2') o OpenAI. En Streamlit Cloud, esta simulación muestra el concepto.")

st.markdown("---")

# ==================== MÓDULO 3: GROQ API ====================
st.header("⚡ Módulo 3: Inferencia y Razonamiento con Groq")

if client:
    # Parámetros de configuración
    col_temp, col_topp, col_model = st.columns(3)
    
    with col_temp:
        temperatura = st.slider("🌡️ Temperatura", 0.0, 1.5, 0.7, 0.05,
                               help="Baja (<0.3): respuestas deterministas. Alta (>0.7): más creatividad/aleatoriedad")
    
    with col_topp:
        top_p = st.slider("🎲 Top-P (Nucleus Sampling)", 0.0, 1.0, 0.9, 0.05,
                         help="Controla la diversidad del vocabulario considerado")
    
    with col_model:
        modelo_groq = st.selectbox("🧠 Modelo", 
                                  ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"])
    
    # System Prompt vs User Prompt
    st.markdown("### 📝 Configuración del Contexto (System Prompt)")
    system_prompt = st.text_area("System Prompt (define el comportamiento del asistente):",
                                "Eres un asistente útil, conciso y experto en tecnología.")
    
    user_prompt = st.text_area("User Prompt (tu pregunta):",
                              "Explica qué es el mecanismo de Self-Attention en los Transformers")
    
    if st.button("🚀 Ejecutar inferencia", type="primary"):
        try:
            # Medición de tiempo
            start_time = time.time()
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = client.chat.completions.create(
                model=modelo_groq,
                messages=messages,
                temperature=temperatura,
                top_p=top_p,
                max_tokens=500
            )
            
            end_time = time.time()
            
            # Métricas de rendimiento
            respuesta_texto = response.choices[0].message.content
            tokens_prompt = response.usage.prompt_tokens
            tokens_completion = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            tiempo_total = end_time - start_time
            time_per_token = (tiempo_total / tokens_completion) * 1000 if tokens_completion > 0 else 0
            throughput = tokens_completion / tiempo_total if tiempo_total > 0 else 0
            
            # Mostrar respuesta
            st.markdown("### 💬 Respuesta del modelo:")
            st.write(respuesta_texto)
            
            # Métricas de desempeño
            st.markdown("### 📊 Métricas de Desempeño")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("⏱️ Tiempo total", f"{tiempo_total:.2f}s")
            with col2:
                st.metric("⚡ Time per Token", f"{time_per_token:.2f} ms")
            with col3:
                st.metric("🚀 Throughput", f"{throughput:.1f} tokens/s")
            with col4:
                st.metric("📥 Tokens input", tokens_prompt)
            with col5:
                st.metric("📤 Tokens output", tokens_completion)
            
            # Explicación didáctica de los parámetros
            with st.expander("📖 Explicación de los parámetros usados"):
                st.markdown(f"""
                - **Temperatura ({temperatura})**: {"Baja → respuestas más deterministas y consistentes" if temperatura < 0.3 else "Alta → respuestas más creativas y diversas" if temperatura > 0.7 else "Media → balance entre creatividad y consistencia"}
                - **Top-P ({top_p})**: {"Bajo → vocabulario más restringido y predecible" if top_p < 0.5 else "Alto → mayor diversidad de palabras posibles"}
                - **Modelo**: {modelo_groq} - {'8B parámetros (rápido)' if '8b' in modelo_groq else '70B parámetros (más preciso)' if '70b' in modelo_groq else 'Mixtral 8x7B (balance)' if 'mixtral' in modelo_groq else 'Gemma 9B (eficiente)'}
                """)
                
        except Exception as e:
            st.error(f"Error al llamar a Groq API: {e}")
else:
    st.error("🔑 Configura tu API Key de Groq en los secrets de Streamlit Cloud para usar este módulo.")

st.markdown("---")

# ==================== MÓDULO 4: ANÁLISIS DE SELF-ATTENTION ====================
st.header("🔄 Módulo 4: Análisis de Self-Attention")
st.markdown("Cambia el contexto y observa cómo el modelo ajusta su atención.")

ejemplo_contexto = st.selectbox("Selecciona un ejemplo de cambio de contexto:",
                               ["Contexto técnico", "Contexto casual", "Contexto histórico"])

if ejemplo_contexto == "Contexto técnico":
    contexto = "Eres un ingeniero experto en Transformers explicando a estudiantes."
    pregunta = "¿Cómo funciona la atención multicabeza?"
elif ejemplo_contexto == "Contexto casual":
    contexto = "Eres un amigo explicando conceptos complejos de forma sencilla."
    pregunta = "Explícame lo de la atención en las redes neuronales como si tuviera 12 años."
else:
    contexto = "Eres un profesor de historia explicando la evolución de la IA."
    pregunta = "¿Cómo se relaciona el concepto de 'atención' con la forma en que los humanos procesamos información?"

if st.button("🧠 Probar Self-Attention con diferente contexto"):
    if client:
        try:
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": contexto},
                    {"role": "user", "content": pregunta}
                ],
                temperature=0.5,
                max_tokens=400
            )
            st.markdown("**Respuesta del modelo según el contexto:**")
            st.write(response.choices[0].message.content)
            st.success("✨ Observa cómo el mismo concepto base (Self-Attention) se explica de forma diferente según el contexto provisto en el System Prompt.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Configura Groq API Key para ver este ejemplo.")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("### 📚 Referencias y notas")
st.markdown("""
- **Self-Attention**: Permite al modelo pesar la importancia de cada token respecto a los demás.
- **Groq Cloud**: Infraestructura de inferencia ultrarrápida (hasta 500+ tokens/s).
- **Modelos disponibles**: Llama 3 (8B/70B), Mixtral 8x7B, Gemma 2 (9B).
- **Para producción real**: Reemplazar embeddings simulados con `sentence-transformers` o `OpenAI embeddings`.
""")