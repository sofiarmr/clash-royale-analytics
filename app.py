import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from textblob import TextBlob

import requests
from bs4 import BeautifulSoup

from PIL import Image

import json
import re
import ast
import unicodedata


# ======================
# CONFIGURACIÓN
# ======================

logo = Image.open("assets/logo.jpg")

st.set_page_config(
    page_title="Clash Royale Analytics",
    page_icon=logo,
    layout="wide"
)



# ======================
# CSS
# ======================

with open("style.css", encoding="utf-8") as f:
    css = f.read()

st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True
)


# ======================
# FUNCIONES PDF
# ======================

def mostrar_pdf(ruta_pdf):
    with open(ruta_pdf, "rb") as archivo_pdf:
        base64_pdf = base64.b64encode(
            archivo_pdf.read()
        ).decode("utf-8")

    st.markdown(
        f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="850px"
            style="border:4px solid #dbe7ff; border-radius:20px; box-shadow:0 10px 24px rgba(0,0,0,.14);">
        </iframe>
        """,
        unsafe_allow_html=True
    )



# ======================
# FUNCIONES AUXILIARES
# ======================

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join([c for c in texto if not unicodedata.combining(c)])
    return texto


def obtener_icono(valor):
    try:
        if isinstance(valor, dict):
            return valor.get("medium", None)

        if isinstance(valor, str):
            try:
                data = ast.literal_eval(valor)
                if isinstance(data, dict):
                    return data.get("medium", None)
            except:
                pass

            if valor.startswith("http"):
                return valor

    except:
        return None

    return None


def analizar_sentimiento(texto):
    texto_limpio = limpiar_texto(texto)

    positivas = [
        "bueno", "excelente", "genial", "increible", "fuerte",
        "util", "me gusta", "recomiendo", "mejor", "divertido",
        "perfecto", "poderosa", "poderoso", "balanceada"
    ]

    negativas = [
        "malo", "pesimo", "debil", "horrible", "odio",
        "nerf", "rota", "roto", "desbalanceada", "desbalanceado",
        "aburrido", "inutil", "peor", "molesta", "molesto"
    ]

    score = 0

    for palabra in positivas:
        if palabra in texto_limpio:
            score += 1

    for palabra in negativas:
        if palabra in texto_limpio:
            score -= 1

    try:
        polaridad_blob = TextBlob(texto).sentiment.polarity
    except:
        polaridad_blob = 0

    polaridad = score + polaridad_blob

    if polaridad > 0:
        return "Positivo", round(polaridad, 3)

    elif polaridad < 0:
        return "Negativo", round(polaridad, 3)

    else:
        return "Neutral", round(polaridad, 3)


def extraer_opiniones(url, limite=25):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    respuesta = requests.get(
        url,
        headers=headers,
        timeout=10
    )

    soup = BeautifulSoup(
        respuesta.text,
        "html.parser"
    )

    textos = []

    etiquetas = soup.find_all(
        ["p", "span", "div", "li"]
    )

    for etiqueta in etiquetas:
        texto = etiqueta.get_text(
            " ",
            strip=True
        )

        if len(texto) >= 40 and len(texto) <= 300:
            textos.append(texto)

    textos_limpios = []

    for t in textos:
        if t not in textos_limpios:
            textos_limpios.append(t)

    return textos_limpios[:limite]


def responder_ia(pregunta, df):
    p = limpiar_texto(pregunta)

    if "fila" in p or "registro" in p or "cuantas cartas" in p or "cantidad de cartas" in p:
        return f"El dataset contiene {df.shape[0]} cartas o registros."

    if "columna" in p or "variables" in p:
        return f"El dataset contiene {df.shape[1]} columnas: {', '.join(df.columns)}."

    if "promedio" in p or "media" in p:
        for col in df.select_dtypes(include=np.number).columns:
            if limpiar_texto(col) in p:
                return f"La media de {col} es {round(df[col].mean(), 3)}."

        if "elixir" in p:
            return f"El costo promedio de elixir es {round(df['elixirCost'].mean(), 2)}."

        if "uso" in p or "usage" in p:
            return f"El uso promedio de las cartas es {round(df['usage'].mean(), 2)}."

        if "vida" in p or "hitpoints" in p:
            return f"El promedio de hitpoints es {round(df['hitpoints'].mean(), 2)}."

    if "maximo" in p or "mayor" in p or "mas alto" in p:
        if "hitpoints" in p or "vida" in p:
            fila = df.sort_values("hitpoints", ascending=False).iloc[0]
            return f"La carta con más hitpoints es {fila['name']} con {fila['hitpoints']} puntos de vida."

        if "uso" in p or "usage" in p or "usada" in p:
            fila = df.sort_values("usage", ascending=False).iloc[0]
            return f"La carta más usada es {fila['name']} con {fila['usage']}% de uso."

        if "elixir" in p:
            fila = df.sort_values("elixirCost", ascending=False).iloc[0]
            return f"La carta con mayor costo de elixir es {fila['name']} con {fila['elixirCost']} de elixir."

    if "minimo" in p or "menor" in p or "mas bajo" in p:
        if "hitpoints" in p or "vida" in p:
            fila = df.sort_values("hitpoints", ascending=True).iloc[0]
            return f"La carta con menos hitpoints es {fila['name']} con {fila['hitpoints']} puntos de vida."

        if "uso" in p or "usage" in p:
            fila = df.sort_values("usage", ascending=True).iloc[0]
            return f"La carta menos usada es {fila['name']} con {fila['usage']}% de uso."

        if "elixir" in p:
            fila = df.sort_values("elixirCost", ascending=True).iloc[0]
            return f"La carta con menor costo de elixir es {fila['name']} con {fila['elixirCost']} de elixir."

    if "legendaria" in p or "legendary" in p:
        total = len(df[df["rarity"].astype(str).str.contains("legendary", case=False, na=False)])
        return f"Hay {total} cartas legendarias."

    if "epica" in p or "epic" in p:
        total = len(df[df["rarity"].astype(str).str.contains("epic", case=False, na=False)])
        return f"Hay {total} cartas épicas."

    if "rara" in p or "rare" in p:
        total = len(df[df["rarity"].astype(str).str.contains("rare", case=False, na=False)])
        return f"Hay {total} cartas raras."

    if "comun" in p or "common" in p:
        total = len(df[df["rarity"].astype(str).str.contains("common", case=False, na=False)])
        return f"Hay {total} cartas comunes."

    if "campeon" in p or "champion" in p:
        total = len(df[df["rarity"].astype(str).str.contains("champion", case=False, na=False)])
        return f"Hay {total} cartas campeonas."

    if "tropa" in p or "troop" in p:
        total = len(df[df["type"].astype(str).str.contains("troop", case=False, na=False)])
        return f"Hay {total} cartas de tipo tropa."

    if "hechizo" in p or "spell" in p:
        total = len(df[df["type"].astype(str).str.contains("spell", case=False, na=False)])
        return f"Hay {total} cartas de tipo hechizo."

    if "estructura" in p or "building" in p:
        total = len(df[df["type"].astype(str).str.contains("building", case=False, na=False)])
        return f"Hay {total} cartas de tipo estructura."

    if "rareza mas frecuente" in p or "rareza predominante" in p:
        rareza = df["rarity"].value_counts().idxmax()
        total = df["rarity"].value_counts().max()
        return f"La rareza más frecuente es {rareza}, con {total} cartas."

    if "tipo mas frecuente" in p or "tipo predominante" in p:
        tipo = df["type"].value_counts().idxmax()
        total = df["type"].value_counts().max()
        return f"El tipo de carta más frecuente es {tipo}, con {total} cartas."

    if "correlacion" in p and "elixir" in p and ("hitpoints" in p or "vida" in p):
        corr = df[["elixirCost", "hitpoints"]].corr().iloc[0, 1]
        return f"La correlación entre costo de elixir y hitpoints es {round(corr, 3)}."

    if "nulos" in p or "faltantes" in p:
        nulos = df.isnull().sum()
        total = nulos[nulos > 0]

        if len(total) == 0:
            return "El dataset no presenta valores nulos."

        return "Valores nulos encontrados:\n" + total.to_string()

    return (
        "No logré interpretar completamente la pregunta. "
        "Puedes preguntar por cantidad de cartas, columnas, promedio de elixir, "
        "carta más usada, carta con más hitpoints, rarezas, tipos de carta o valores máximos y mínimos."
    )


# ======================
# CARGA DATASET
# ======================

@st.cache_data
def load_data():

    try:
        df = pd.read_json(
            "data/clash_royale_cards.json"
        )

        if "items" in df.columns:
            df = pd.json_normalize(df["items"])

    except:
        df = pd.read_excel(
            "data/clash_royale_cards.xlsx"
        )

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return df


df = load_data()


# ======================
# MENÚ RESPONSIVE
# ======================

opciones = {
    "Inicio": "Inicio",
    "Exploración": "Exploración",
    "Machine Learning": "Machine Learning",
    "Recomendador": "Recomendador",
    "Carga Archivos": "Carga Archivos",
    "Sentimientos": "Sentimientos",
    "Asistente IA": "Asistente IA",
    "Manual de Usuario": "Manual de Usuario",
    "Acerca": "Acerca"
}

if "page" not in st.query_params:
    st.query_params["page"] = "Inicio"

menu = st.query_params["page"]

if menu not in opciones:
    menu = "Inicio"


# ======================
# MENÚ SUPERIOR PARA CELULAR
# ======================

mobile_nav = st.container(key="mobile_nav")

with mobile_nav:
    with st.popover("☰ Menú"):
        for key, label in opciones.items():
            if st.button(
                label,
                key=f"mobile_btn_{key}"
            ):
                st.query_params["page"] = key
                st.rerun()


# ======================
# SIDEBAR PARA ESCRITORIO
# ======================

with st.sidebar:

    c1, c2, c3 = st.columns([1,2,1])

    with c2:
        st.image(
            logo,
            width=150
        )

st.sidebar.markdown(
    """
    <h2 style='text-align:center;color:white'>
    Clash Royale Analytics
    </h2>
    """,
    unsafe_allow_html=True
)

for key, label in opciones.items():

    clase = "menu-button-active" if menu == key else "menu-button"

    st.sidebar.markdown(
        f"""
        <div class="{clase}">
            {label}
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.sidebar.button(
        label,
        key=f"sidebar_btn_{key}"
    ):
        st.query_params["page"] = key
        st.rerun()

# =====================================
# INICIO
# =====================================

if menu == "Inicio":

    imagenes = [
        "assets/banner.jpg",
        "assets/banner1.jpg",
        "assets/banner2.jpg",
        "assets/banner3.jpg",
        "assets/banner4.jpg"
    ]

    if "banner_actual" not in st.session_state:
        st.session_state.banner_actual = 0

    st.image(
        imagenes[st.session_state.banner_actual],
        use_container_width=True
    )

    col1, col2, col3 = st.columns([5, 2, 5])

    with col2:

        b1, b2 = st.columns(2)

        with b1:
            if st.button("◀", key="banner_prev"):
                st.session_state.banner_actual = (
                    st.session_state.banner_actual - 1
                ) % len(imagenes)
                st.rerun()

        with b2:
            if st.button("▶", key="banner_next"):
                st.session_state.banner_actual = (
                    st.session_state.banner_actual + 1
                ) % len(imagenes)
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    col_foto, col_info = st.columns([1, 2])

    with col_foto:

        st.image(
            "assets/foto.jpeg",
            width=260
        )

    with col_info:

        with col_info:

            st.markdown(
        """
        <div class="info-box">

        <h2>👩‍💻 SOFÍA MARGARITA ROMERO RODRÍGUEZ</h2>

        <span>
        Estudiante de Ingeniería en Sistemas y Redes Informáticas con interés en Ciencia de Datos,
        desarrollo de software, diseño de interfaces y análisis de información.
        <br><br>
        Este portafolio presenta una aplicación interactiva basada en datos de Clash Royale,
        integrando visualización, machine learning, recomendación, scraping e inteligencia artificial.
        </span>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "📊 Cartas",
            len(df)
        )

    with c2:
        st.metric(
            "🧬 Variables",
            len(df.columns)
        )

    with c3:
        st.metric(
            "🎴 Rarezas",
            df["rarity"].nunique()
        )

    with c4:
        st.metric(
            "⚔️ Tipos",
            df["type"].nunique()
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
    """
<div class="info-box">

<h2>🎬 Storytelling</h2>

<span>
En este video se presenta una demostración del análisis realizado a partir del dataset de cartas de Clash Royale, explicando los hallazgos principales del análisis exploratorio, las hipótesis planteadas y los resultados obtenidos mediante modelos de Machine Learning.
</span>

</div>
    """,
    unsafe_allow_html=True
)

    st.video(
        "https://youtu.be/luSodC9KUp4?feature=shared"
    )

    st.markdown("<br>", unsafe_allow_html=True)


# =====================================
# EXPLORACIÓN
# =====================================

elif menu == "Exploración":

    st.title(
        "📊 Exploración de Datos"
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Descripción",
        "Dataset",
        "Variables",
        "Navegador",
        "Buscador",
        "Graficador",
        "Hipótesis"
    ])

    with tab1:

        st.subheader(
            "Descripción general del dataset"
        )

        st.markdown(
            """
            El dataset utilizado contiene información sobre cartas del videojuego
            **Clash Royale**.

            Cada fila representa una carta del juego y cada columna describe
            una característica relevante de dicha carta, como su rareza,
            costo de elixir, tipo, movilidad, objetivos, puntos de vida
            y porcentaje de uso.

            Este conjunto de datos permite realizar análisis exploratorio,
            formular hipótesis, entrenar modelos predictivos y crear un
            sistema de recomendación basado en similitud entre cartas.
            """
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Filas",
                df.shape[0]
            )

        with c2:
            st.metric(
                "Columnas",
                df.shape[1]
            )

        with c3:
            st.metric(
                "Tipos de carta",
                df["type"].nunique()
            )

        with c4:
            st.metric(
                "Rarezas",
                df["rarity"].nunique()
            )

        st.subheader(
            "Distribución de cartas por tipo"
        )

        tipo_df = (
            df["type"]
            .value_counts()
            .reset_index()
        )

        tipo_df.columns = [
            "Tipo",
            "Cantidad"
        ]

        fig = px.pie(
            tipo_df,
            names="Tipo",
            values="Cantidad",
            title="Distribución por tipo de carta"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Distribución de cartas por rareza"
        )

        rareza_df = (
            df["rarity"]
            .value_counts()
            .reset_index()
        )

        rareza_df.columns = [
            "Rareza",
            "Cantidad"
        ]

        fig2 = px.bar(
            rareza_df,
            x="Rareza",
            y="Cantidad",
            title="Cantidad de cartas por rareza"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    with tab2:

        st.subheader("Vista General del Dataset")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.write(
            "Filas:",
            df.shape[0]
        )

        st.write(
            "Columnas:",
            df.shape[1]
        )

        st.subheader(
            "Resumen estadístico"
        )

        st.dataframe(
            df.describe(
                include="all"
            ),
            use_container_width=True
        )

        st.subheader(
            "Valores nulos"
        )

        nulos = (
            df.isnull()
            .sum()
            .reset_index()
        )

        nulos.columns = [
            "Campo",
            "Valores nulos"
        ]

        st.dataframe(
            nulos,
            use_container_width=True
        )

    with tab3:

        st.subheader(
            "Descripción de campos"
        )

        descripciones = {

            "name":
            "Nombre oficial de la carta dentro del juego Clash Royale.",

            "maxEvolutionLevel":
            "Nivel máximo de evolución disponible para la carta. Puede estar vacío si no posee evolución.",

            "elixirCost":
            "Cantidad de elixir necesaria para desplegar la carta durante una partida.",

            "iconUrls":
            "Dirección o estructura que contiene la imagen visual de la carta.",

            "evolutionIcons":
            "Dirección o estructura que contiene la imagen de evolución de la carta, si aplica.",

            "rarity":
            "Rareza de la carta. Puede ser common, rare, epic, legendary o champion.",

            "type":
            "Tipo general de carta. Puede ser troop, spell o building.",

            "mobility":
            "Indica si la carta se desplaza por tierra o aire.",

            "targets":
            "Tipo de objetivo que puede atacar la carta: ground, both o buildings.",

            "attack_type":
            "Tipo de ataque de la carta: single o splash.",

            "groupCard":
            "Indica si la carta representa una unidad individual o un grupo de unidades.",

            "usage":
            "Porcentaje de uso de la carta dentro del contexto del dataset.",

            "hitpoints":
            "Cantidad de puntos de vida de la carta.",

            "hp_level":
            "Clasificación categórica del nivel de vida de la carta: low, medium o high."
        }

        variable = st.selectbox(
            "Seleccione variable",
            df.columns
        )

        st.info(
            descripciones.get(
                variable,
                "No hay descripción registrada para esta variable."
            )
        )

        st.write(
            "Tipo de dato:",
            df[variable].dtype
        )

        if pd.api.types.is_numeric_dtype(
            df[variable]
        ):

            st.subheader(
                "Medidas estadísticas"
            )

            st.dataframe(
                df[variable]
                .describe()
                .to_frame(),
                use_container_width=True
            )

            fig = px.histogram(
                df,
                x=variable,
                title=f"Distribución de {variable}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.subheader(
                "Valores posibles"
            )

            valores = (
                df[variable]
                .dropna()
                .astype(str)
                .unique()
            )

            st.write(
                valores
            )

            conteo = (
                df[variable]
                .astype(str)
                .value_counts()
                .reset_index()
            )

            conteo.columns = [
                variable,
                "Cantidad"
            ]

            st.dataframe(
                conteo,
                use_container_width=True
            )

    with tab4:

        st.subheader(
            "Navegador del dataset completo"
        )

        st.dataframe(
            df,
            use_container_width=True,
            height=600
        )

    with tab5:

        st.subheader(
            "Buscador de cartas"
        )

        if "name" in df.columns:

            nombre = st.text_input(
                "Buscar carta por nombre"
            )

            resultado = df[
                df["name"]
                .astype(str)
                .str.contains(
                    nombre,
                    case=False,
                    na=False
                )
            ]

            st.dataframe(
                resultado,
                use_container_width=True
            )

    with tab6:

        st.subheader(
            "Graficador exploratorio"
        )

        columna = st.selectbox(
            "Seleccione variable para graficar",
            df.columns
        )

        if pd.api.types.is_numeric_dtype(
            df[columna]
        ):

            fig = px.histogram(
                df,
                x=columna,
                title=f"Histograma de {columna}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            conteo = (
                df[columna]
                .astype(str)
                .value_counts()
                .reset_index()
            )

            conteo.columns = [
                columna,
                "Cantidad"
            ]

            fig = px.bar(
                conteo,
                x=columna,
                y="Cantidad",
                title=f"Distribución de {columna}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with tab7:

        st.subheader(
            "Análisis de Hipótesis"
        )

        hipotesis = st.selectbox(
            "Seleccione una hipótesis",
            [
                "Las cartas con mayor costo de elixir poseen más hitpoints",
                "Las cartas aéreas tienen mayor uso promedio que las terrestres",
                "Las cartas grupales poseen menos hitpoints promedio que las cartas individuales"
            ]
        )

        if hipotesis == "Las cartas con mayor costo de elixir poseen más hitpoints":

            st.markdown(
                """
                **Hipótesis:** A mayor costo de elixir, mayor cantidad de puntos de vida.
                """
            )

            fig = px.scatter(
                df,
                x="elixirCost",
                y="hitpoints",
                color="rarity",
                hover_name="name",
                title="Costo de Elixir vs Hitpoints"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            correlacion = df[
                ["elixirCost", "hitpoints"]
            ].corr().iloc[0, 1]

            st.metric(
                "Correlación",
                round(correlacion, 3)
            )

            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                 "🏆 Carta más resistente",
                "Golem"
            )

            with c2:
                st.metric(
                 "❤️ Mayor HP",
                "5100"
            )

            with c3:
                st.metric(
                 "📈 Correlación",
                "0.662"
            )

            if correlacion > 0:
                st.success(
                    "Conclusión: La hipótesis se acepta parcialmente, ya que existe una relación positiva entre el costo de elixir y los puntos de vida."
                )
            else:
                st.error(
                    "Conclusión: La hipótesis no se acepta, ya que no existe una relación positiva entre ambas variables."
                )

        elif hipotesis == "Las cartas aéreas tienen mayor uso promedio que las terrestres":

            st.markdown(
                """
                **Hipótesis:** Las cartas con movilidad aérea tienen un uso promedio mayor que las cartas terrestres.
                """
            )

            promedio = (
                df.groupby("mobility")["usage"]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                promedio,
                x="mobility",
                y="usage",
                title="Uso promedio por movilidad"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                promedio,
                use_container_width=True
            )

            mayor = promedio.sort_values(
                "usage",
                ascending=False
            ).iloc[0]

            st.info(
                f"Conclusión: La movilidad con mayor uso promedio es {mayor['mobility']} con un promedio de {round(mayor['usage'], 2)}."
            )

        else:

            st.markdown(
                """
                **Hipótesis:** Las cartas grupales poseen menos hitpoints promedio que las cartas individuales.
                """
            )

            promedio = (
                df.groupby("groupCard")["hitpoints"]
                .mean()
                .reset_index()
            )

            promedio["Tipo"] = promedio["groupCard"].map({
                True: "Carta grupal",
                False: "Carta individual"
            })

            fig = px.bar(
                promedio,
                x="Tipo",
                y="hitpoints",
                title="Hitpoints promedio según tipo de carta grupal"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                promedio[["Tipo", "hitpoints"]],
                use_container_width=True
            )

            st.info(
                "Conclusión: Se compara el promedio de puntos de vida entre cartas grupales e individuales para validar si las cartas de grupo tienden a tener menor resistencia."
            )


# =====================================
# MACHINE LEARNING
# =====================================

elif menu == "Machine Learning":

    st.title("🤖 Machine Learning")

    objetivos = [
        col for col in [
            "usage",
            "hitpoints"
        ]
        if col in df.columns
    ]

    objetivo = st.selectbox(
        "Variable a predecir",
        objetivos
    )

    posibles = [
        "elixirCost",
        "rarity",
        "type",
        "mobility",
        "targets",
        "attack_type",
        "hp_level",
        "groupCard"
    ]

    columnas_disponibles = [
        c for c in posibles
        if c in df.columns and c != objetivo
    ]

    columnas = st.multiselect(
        "Variables independientes",
        columnas_disponibles,
        default=columnas_disponibles[:4]
    )

    porcentaje = st.slider(
        "Porcentaje entrenamiento",
        60,
        90,
        80
    )

    train_size = porcentaje / 100

    modelo = st.selectbox(
        "Modelo",
        [
            "Regresión Lineal",
            "Random Forest"
        ]
    )

    if len(columnas) == 0:
        st.warning(
            "Seleccione al menos una variable independiente."
        )

    else:

        data_ml = df[
            columnas + [objetivo]
        ].copy()

        data_ml = data_ml.dropna()

        X = pd.get_dummies(
            data_ml[columnas],
            drop_first=True
        )

        y = data_ml[objetivo]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            random_state=42
        )

        if st.button("Entrenar Modelo"):

            if modelo == "Regresión Lineal":

                model = LinearRegression()

            else:

                model = RandomForestRegressor(
                    n_estimators=200,
                    random_state=42
                )

            model.fit(
                X_train,
                y_train
            )

            pred_test = model.predict(
                X_test
            )

            pred_train = model.predict(
                X_train
            )

            r2 = r2_score(
                y_test,
                pred_test
            )

            mae = mean_absolute_error(
                y_test,
                pred_test
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    pred_test
                )
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "R²",
                    round(r2, 3)
                )

            with c2:
                st.metric(
                    "MAE",
                    round(mae, 3)
                )

            with c3:
                st.metric(
                    "RMSE",
                    round(rmse, 3)
                )

            resultado_train = pd.DataFrame(
                {
                    "Real": y_train,
                    "Predicción": pred_train,
                    "Tipo": "Entrenamiento"
                }
            )

            resultado_test = pd.DataFrame(
                {
                    "Real": y_test,
                    "Predicción": pred_test,
                    "Tipo": "Prueba"
                }
            )

            resultado = pd.concat(
                [
                    resultado_train,
                    resultado_test
                ]
            )

            fig = px.scatter(
                resultado,
                x="Real",
                y="Predicción",
                color="Tipo",
                title=f"Predicción usando {modelo}"
            )

            fig.add_shape(
                type="line",
                x0=resultado["Real"].min(),
                y0=resultado["Real"].min(),
                x1=resultado["Real"].max(),
                y1=resultado["Real"].max()
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader(
                "Comparación de resultados de prueba"
            )

            st.dataframe(
                resultado_test.head(20),
                use_container_width=True
            )

            if modelo == "Regresión Lineal":

                coeficientes = pd.DataFrame(
                    {
                        "Variable": X.columns,
                        "Coeficiente": model.coef_
                    }
                )

                st.subheader(
                    "Coeficientes del modelo"
                )

                st.dataframe(
                    coeficientes,
                    use_container_width=True
                )

            if modelo == "Random Forest":

                importancia = pd.DataFrame(
                    {
                        "Variable": X.columns,
                        "Importancia": model.feature_importances_
                    }
                )

                importancia = importancia.sort_values(
                    "Importancia",
                    ascending=False
                )

                fig2 = px.bar(
                    importancia.head(10),
                    x="Importancia",
                    y="Variable",
                    orientation="h",
                    title="Variables más importantes"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )


# =====================================
# RECOMENDADOR
# =====================================

elif menu == "Recomendador":

    st.title("🎮 Sistema de Recomendación")

    st.write(
        "Seleccione una carta y el sistema buscará otras cartas similares usando características como rareza, tipo, movilidad, objetivos y estilo de ataque."
    )

    columnas_rec = []

    posibles = [
        "rarity",
        "type",
        "mobility",
        "targets",
        "attack_type",
        "hp_level",
        "groupCard"
    ]

    for c in posibles:
        if c in df.columns:
            columnas_rec.append(c)

    datos = df.copy()

    datos[columnas_rec] = (
        datos[columnas_rec]
        .fillna("Desconocido")
    )

    matriz = pd.get_dummies(
        datos[columnas_rec]
    )

    similitud = cosine_similarity(
        matriz
    )

    carta = st.selectbox(
        "Seleccione una carta",
        sorted(df["name"].astype(str).unique())
    )

    indice = (
        df[
            df["name"] == carta
        ]
        .index[0]
    )

    carta_base = df.iloc[indice]

    st.subheader(
        "Carta seleccionada"
    )

    col_img, col_info = st.columns([1, 3])

    with col_img:

        icono = obtener_icono(
            carta_base.get("iconUrls", None)
        )

        if icono:
            st.image(
                icono,
                width=150
            )

    with col_info:

        st.markdown(
            f"""
            ### {carta_base['name']}

            - **Rareza:** {carta_base.get('rarity', 'N/A')}
            - **Tipo:** {carta_base.get('type', 'N/A')}
            - **Costo de elixir:** {carta_base.get('elixirCost', 'N/A')}
            - **Uso:** {carta_base.get('usage', 'N/A')}
            - **Hitpoints:** {carta_base.get('hitpoints', 'N/A')}
            """
        )

    scores = list(
        enumerate(
            similitud[indice]
        )
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    recomendaciones = scores[1:6]

    st.subheader(
        "Cartas recomendadas"
    )

    cols = st.columns(5)

    for i, rec in enumerate(
        recomendaciones
    ):

        carta_rec = df.iloc[
            rec[0]
        ]

        with cols[i]:

            icono = obtener_icono(
                carta_rec.get("iconUrls", None)
            )

            if icono:
                st.image(
                    icono,
                    use_container_width=True
                )

            st.success(
                carta_rec["name"]
            )

            st.caption(
                f"Similitud: {round(rec[1], 2)}"
            )

            st.write(
                f"Rareza: {carta_rec.get('rarity', 'N/A')}"
            )

            st.write(
                f"Elixir: {carta_rec.get('elixirCost', 'N/A')}"
            )


# =====================================
# CARGA ARCHIVOS
# =====================================

elif menu == "Carga Archivos":

    st.title("📂 Carga de Archivos")

    archivo = st.file_uploader(
        "Suba un archivo",
        type=[
            "csv",
            "xlsx",
            "json"
        ]
    )

    if archivo:

        if archivo.name.endswith(
            ".csv"
        ):
            nuevo_df = pd.read_csv(
                archivo
            )

        elif archivo.name.endswith(
            ".xlsx"
        ):
            nuevo_df = pd.read_excel(
                archivo
            )

        else:
            nuevo_df = pd.read_json(
                archivo
            )

        st.success(
            "Archivo cargado correctamente"
        )

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Filas",
                nuevo_df.shape[0]
            )

        with c2:
            st.metric(
                "Columnas",
                nuevo_df.shape[1]
            )

        st.subheader(
            "Vista previa"
        )

        st.dataframe(
            nuevo_df.head(),
            use_container_width=True
        )

        st.subheader(
            "Resumen estadístico"
        )

        st.dataframe(
            nuevo_df.describe(
                include="all"
            ),
            use_container_width=True
        )

        numericas = (
            nuevo_df
            .select_dtypes(
                include=np.number
            )
            .columns
        )

        if len(numericas) > 0:

            variable = st.selectbox(
                "Variable numérica para graficar",
                numericas
            )

            fig = px.histogram(
                nuevo_df,
                x=variable,
                title=f"Distribución de {variable}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =====================================
# SENTIMIENTOS + SCRAPING
# =====================================

elif menu == "Sentimientos":

    st.title(
        "😊 Análisis de Sentimientos y Scraping"
    )

    st.markdown(
        """
        Esta sección permite extraer textos de una página web mediante
        scraping y luego analizar el sentimiento de las opiniones encontradas.
        """
    )

    url = st.text_input(
        "Ingrese la URL de una página con opiniones o comentarios"
    )

    usar_ejemplo = st.checkbox(
        "Usar opiniones de ejemplo sobre Clash Royale",
        value=True
    )

    opiniones = []

    if st.button(
        "Extraer y analizar opiniones"
    ):

        if usar_ejemplo or url.strip() == "":

            opiniones = [
                "El Montapuercos es una carta muy fuerte y divertida de usar.",
                "El Mega Caballero está demasiado roto y arruina muchas partidas.",
                "La Valquiria es bastante útil para defender tropas terrestres.",
                "El Mago eléctrico me parece una carta excelente contra unidades rápidas.",
                "El Golem es muy lento pero puede ser poderoso si se usa bien.",
                "No me gusta jugar contra el Barril de Duendes porque es molesto.",
                "La Princesa es una carta legendaria muy buena por su largo alcance.",
                "El Espíritu de Hielo es barato y muy útil para ciclar mazos.",
                "Algunas cartas necesitan un nerf porque están desbalanceadas.",
                "Clash Royale sigue siendo un juego entretenido y estratégico."
            ]

        else:

            try:
                opiniones = extraer_opiniones(
                    url
                )

                if len(opiniones) == 0:
                    st.warning(
                        "No se encontraron textos suficientes en la página. Se usarán opiniones de ejemplo."
                    )

                    opiniones = [
                        "Clash Royale es un juego muy entretenido.",
                        "Algunas cartas parecen demasiado fuertes.",
                        "El balance del juego puede mejorar.",
                        "Me gusta usar cartas rápidas y baratas.",
                        "No me gusta cuando una carta está rota."
                    ]

            except Exception as e:
                st.error(
                    f"No se pudo realizar el scraping: {e}"
                )

                opiniones = [
                    "Clash Royale es un juego muy entretenido.",
                    "Algunas cartas parecen demasiado fuertes.",
                    "El balance del juego puede mejorar.",
                    "Me gusta usar cartas rápidas y baratas.",
                    "No me gusta cuando una carta está rota."
                ]

        resultados = []

        for opinion in opiniones:

            sentimiento, polaridad = analizar_sentimiento(
                opinion
            )

            resultados.append(
                {
                    "Opinión": opinion,
                    "Sentimiento": sentimiento,
                    "Polaridad": polaridad
                }
            )

        df_sent = pd.DataFrame(
            resultados
        )

        st.subheader(
            "Opiniones analizadas"
        )

        st.dataframe(
            df_sent,
            use_container_width=True
        )

        conteo = (
            df_sent["Sentimiento"]
            .value_counts()
            .reset_index()
        )

        conteo.columns = [
            "Sentimiento",
            "Cantidad"
        ]

        fig = px.pie(
            conteo,
            names="Sentimiento",
            values="Cantidad",
            title="Distribución de sentimientos"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Positivos",
                len(df_sent[df_sent["Sentimiento"] == "Positivo"])
            )

        with c2:
            st.metric(
                "Neutrales",
                len(df_sent[df_sent["Sentimiento"] == "Neutral"])
            )

        with c3:
            st.metric(
                "Negativos",
                len(df_sent[df_sent["Sentimiento"] == "Negativo"])
            )


# =====================================
# ASISTENTE IA
# =====================================

elif menu == "Asistente IA":

    st.title(
        "🧠 Asistente Inteligente"
    )

    st.markdown(
        """
        Realice preguntas sobre el dataset de cartas de Clash Royale.
        """
    )

    st.info(
        """
        Ejemplos:
        - ¿Cuántas cartas hay?
        - ¿Cuántas columnas tiene el dataset?
        - ¿Cuál es el promedio de elixir?
        - ¿Cuál es la carta más usada?
        - ¿Cuál tiene más hitpoints?
        - ¿Cuántas cartas legendarias hay?
        - ¿Cuántas tropas hay?
        - ¿Cuál es la rareza más frecuente?
        """
    )

    pregunta = st.text_input(
        "Haga una pregunta"
    )

    if pregunta:

        respuesta = responder_ia(
            pregunta,
            df
        )

        st.success(
            respuesta
        )



# =====================================
# MANUAL DE USUARIO
# =====================================

elif menu == "Manual de Usuario":

    st.title(
        "📘 Manual de Usuario"
    )

    st.markdown(
        """
        En esta sección se muestra el manual oficial de uso de
        **Clash Royale Analytics Pro**. El documento explica el objetivo
        del sistema, sus módulos principales y la forma correcta de utilizar
        cada apartado de la aplicación.
        """
    )

    ruta_manual = "assets/manual_usuario_clash_royale_analytics.pdf"

    try:
        with open(ruta_manual, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        st.download_button(
            label="📥 Descargar Manual de Usuario",
            data=pdf_bytes,
            file_name="Manual_Usuario_Clash_Royale_Analytics_Pro.pdf",
            mime="application/pdf"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        mostrar_pdf(ruta_manual)

    except FileNotFoundError:
        st.error(
            "No se encontró el archivo del manual. Verifique que el PDF esté guardado en assets/manual_usuario_clash_royale_analytics.pdf"
        )

# =====================================
# ACERCA
# =====================================

elif menu == "Acerca":

    st.title(
        "ℹ️ Acerca del Proyecto"
    )

    st.markdown(
        """
        ### Clash Royale Analytics Pro

        Proyecto desarrollado para la asignatura de Ciencia de Datos.

        Este proyecto integra:

        - Streamlit
        - Pandas
        - Plotly
        - Scikit-Learn
        - Machine Learning
        - Sistema de Recomendación
        - Scraping Web
        - Análisis de Sentimientos
        - Asistente de IA basado en consultas del dataset

        ### Dataset

        Clash Royale Cards Dataset.

        Cada registro representa una carta del juego y contiene información
        como rareza, tipo, costo de elixir, movilidad, puntos de vida y uso.

        ### Autor

        Sofía Romero
        """
    )