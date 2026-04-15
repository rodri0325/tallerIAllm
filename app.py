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

# ==================== MÓDULO 1: EL LABORATORIO DEL TOKENIZADOR (MEJORADO) ====================
st.header("📝 Módulo 1: El Laboratorio del Tokenizador")
st.markdown("Visualiza cómo el texto se convierte en tokens con **colores por tipo** según la teoría de tokenización")

# Teoría de tokenización
with st.expander("📖 Teoría de Tokenización - ¿Por qué diferentes colores?"):
    st.markdown("""
    ### Tipos de tokens según su naturaleza:
    
    - 🔴 **Palabras completas**: Tokens que representan palabras enteras (ej: 'hola', 'mundo')
    - 🟡 **Subpalabras**: Partes de palabras más largas (ej: 'acion', 'mente')
    - 🟢 **Prefijos/Sufijos**: Morfemas gramaticales (ej: 're', 'pre', 'ción')
    - 🔵 **Espacios y puntuación**: Caracteres especiales (ej: ' ', '.', ',', '?')
    - 🟣 **Números**: Dígitos y valores numéricos
    - 🟠 **Tokens especiales**: `<|endoftext|>`, `<s>`, `</s>`
    
    **Los LLMs no ven letras, ven estos tokens coloreados conceptualmente!**
    """)

col1, col2 = st.columns([3, 1])

with col1:
    texto_input = st.text_area(
        "📝 Ingresa tu texto para tokenizar:", 
        "El rey, el hombre y la mujer fueron al castillo. ¡Hola mundo! 12345",
        height=120,
        key="texto_tokenizar"
    )

with col2:
    st.markdown("### 🎨 Leyenda de colores")
    st.markdown("🔴 **Palabras completas**")
    st.markdown("🟡 **Subpalabras**")
    st.markdown("🟢 **Prefijos/Sufijos**")
    st.markdown("🔵 **Espacios/Puntuación**")
    st.markdown("🟣 **Números**")
    st.markdown("🟠 **Tokens especiales**")

# Botón específico para tokenizar
if st.button("🔮 TOKENIZAR", type="primary", use_container_width=True):
    if texto_input:
        with st.spinner("Tokenizando el texto..."):
            # Usar tokenizador de GPT-2/LLaMA
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(texto_input)
            token_strings = [encoding.decode([t]) for t in tokens]
            
            # Métricas
            col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
            with col_metric1:
                st.metric("📏 Caracteres", len(texto_input))
            with col_metric2:
                st.metric("🔢 Tokens totales", len(tokens))
            with col_metric3:
                st.metric("📊 Ratio", f"{len(tokens)/len(texto_input):.2f}")
            with col_metric4:
                st.metric("💾 Compresión", f"{(1 - len(tokens)/len(texto_input))*100:.1f}%")
            
            # Función para determinar el tipo de token y su color
            def get_token_color(token_str, token_id):
                token_lower = token_str.lower()
                
                # Tokens especiales
                if token_id in [100257, 100276, 100277] or '<|' in token_str:
                    return "#FF6B35"  # Naranja para tokens especiales
                
                # Números
                if token_str.isdigit() or (token_str[0] == '-' and token_str[1:].isdigit()):
                    return "#9B59B6"  # Púrpura para números
                
                # Espacios
                if token_str.isspace() or token_str == ' ':
                    return "#3498DB"  # Azul para espacios
                
                # Puntuación
                if token_str in ['.', ',', ';', ':', '!', '?', '¡', '¿', '(', ')', '[', ']', '{', '}', '"', "'", '...']:
                    return "#3498DB"  # Azul para puntuación
                
                # Prefijos comunes (español/inglés)
                prefijos = ['re', 'pre', 'post', 'anti', 'sub', 'super', 'des', 'in', 'im', 'non', 'un']
                sufijos = ['ción', 'sión', 'mente', 'ando', 'iendo', 'ado', 'ido', 'ción', 'tivo', 'able', 'ible']
                
                # Subpalabras (comienzan con espacio o tienen caracteres especiales)
                if token_str.startswith(' ') and len(token_str) > 1:
                    return "#F39C12"  # Amarillo para subpalabras con espacio
                
                # Prefijos o sufijos
                if any(token_lower.startswith(p) for p in prefijos) or any(token_lower.endswith(s) for s in sufijos):
                    return "#2ECC71"  # Verde para prefijos/sufijos
                
                # Subpalabras (contienen '##' o son partes de palabras)
                if '##' in token_str or (len(token_str) > 3 and token_str[0].isalpha() and not token_str[0].isupper()):
                    return "#F1C40F"  # Amarillo para subpalabras
                
                # Palabras completas (por defecto)
                return "#E74C3C"  # Rojo para palabras completas
            
            # Mostrar tokens en formato visual estilo "nube de tokens"
            st.markdown("### 🎨 Visualización de tokens por tipo:")
            
            # Crear filas de tokens (máximo 10 por fila)
            tokens_por_fila = 12
            for i in range(0, len(token_strings), tokens_por_fila):
                cols = st.columns(min(tokens_por_fila, len(token_strings) - i))
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(token_strings):
                        token_str = token_strings[idx]
                        token_id = tokens[idx]
                        color = get_token_color(token_str, token_id)
                        
                        # Escapar caracteres especiales para HTML
                        token_display = token_str.replace(' ', '␣').replace('\n', '↵').replace('\t', '→')
                        
                        # Crear tarjeta de token
                        col.markdown(
                            f"""
                            <div style='
                                background-color: {color};
                                padding: 8px 12px;
                                margin: 4px;
                                border-radius: 8px;
                                text-align: center;
                                font-family: monospace;
                                font-weight: bold;
                                color: white;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                                transition: transform 0.2s;
                                cursor: pointer;
                            '
                            onmouseover="this.style.transform='scale(1.05)'"
                            onmouseout="this.style.transform='scale(1)'">
                                <div style='font-size: 16px;'>'{token_display}'</div>
                                <div style='font-size: 10px; opacity: 0.8;'>ID: {token_id}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            
            # Tabla detallada
            with st.expander("📊 Ver tabla detallada de tokenización"):
                tipos_token = []
                for i, (token_str, token_id) in enumerate(zip(token_strings, tokens)):
                    color = get_token_color(token_str, token_id)
                    tipo = "🔴 Palabra completa"
                    if color == "#F1C40F":
                        tipo = "🟡 Subpalabra"
                    elif color == "#2ECC71":
                        tipo = "🟢 Prefijo/Sufijo"
                    elif color == "#3498DB":
                        tipo = "🔵 Espacio/Puntuación"
                    elif color == "#9B59B6":
                        tipo = "🟣 Número"
                    elif color == "#FF6B35":
                        tipo = "🟠 Token especial"
                    
                    tipos_token.append(tipo)
                
                df_detalle = pd.DataFrame({
                    "Posición": range(len(tokens)),
                    "Token": token_strings,
                    "Token ID": tokens,
                    "Tipo": tipos_token
                })
                st.dataframe(df_detalle, use_container_width=True, height=300)
            
            # Gráfico de composición de tokens
            from collections import Counter
            tipos_count = Counter(tipos_token)
            
            fig = px.pie(
                values=list(tipos_count.values()),
                names=list(tipos_count.keys()),
                title="Composición del texto por tipo de token",
                color_discrete_map={
                    "🔴 Palabra completa": "#E74C3C",
                    "🟡 Subpalabra": "#F1C40F",
                    "🟢 Prefijo/Sufijo": "#2ECC71",
                    "🔵 Espacio/Puntuación": "#3498DB",
                    "🟣 Número": "#9B59B6",
                    "🟠 Token especial": "#FF6B35"
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Explicación didáctica
            with st.expander("💡 ¿Qué significa esto?"):
                st.markdown(f"""
                **Análisis de la tokenización:**
                
                - Tu texto tiene **{len(texto_input)} caracteres** pero el LLM lo ve como **{len(tokens)} tokens**
                - Los modelos de lenguaje **no ven letras individuales**, ven estos bloques de significado
                - **Tokens más comunes**: Palabras frecuentes como 'el', 'la', 'que' son un solo token
                - **Subpalabras**: Palabras largas o raras se dividen en partes más pequeñas
                - **Espacios y puntuación**: Son tokens importantes que dan estructura
                
                🧠 **Esto es crucial**: Cuando le preguntas a un LLM, él está prediciendo el siguiente TOKEN, no la siguiente letra.
                """)
    else:
        st.warning("⚠️ Por favor ingresa algún texto para tokenizar")

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
