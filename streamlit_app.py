# Procesamiento de datos
import numpy as np
import pandas as pd
import altair as alt

# Herramientas de gramática
import language_tool_python
import contractions

# Procesamiento de lenguaje natural
import nltk
from textblob import TextBlob, Word
from textblob.sentiments import PatternAnalyzer, NaiveBayesAnalyzer

# Integración
import streamlit as st
import base64

# Configuro la página

st.set_page_config(page_title = "Evalúa tu inglés ahora!", page_icon = "images/page-icon.png", layout = "wide", initial_sidebar_state = "expanded")

def set_background(image_file):

    with open(image_file, "rb") as image_file:

        encoded_string = base64.b64encode(image_file.read())

    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

set_background('images/background-file.png')

# Título y subtítulo

st.markdown("<h1 style='text-align: center; color: white;'>Evalúa al instante tu nivel de inglés</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: white;'>Con este asistente de escritura gratuito, compruebe si hay errores gramaticales, de estilo y ortográficos en su texto en inglés</h1>", unsafe_allow_html=True)

placeholder = st.empty()

# Creo las columnas

with placeholder.container():

    col1, col2 = st.columns(2)

    text = col1.text_area("Texto original:", height = 700, max_chars = None, key = "init_original_text", help = "Introduce en cuadro el texto que desees comprobar", placeholder = "Introduce aquí tu texto")

    col2.text_area("Texto corregido:", height = 700, max_chars = None, key = "init_corrected_text", help = "En este cuadro se mostrará el texto corregido", placeholder = "Aquí se mostrarán las correciones", disabled = True)

    button = st.button('Corregir texto', key = "init_correct_button")
    
# Streamlit
    
if button == True:
    
    if text == "":
    
        st.info('Introduce un texto de ejemplo')

    else:

        with st.spinner("Espera mientras se corrige y evalúa tu texto..."):

            # Errores ortográficos y tipográficos 

            tool = language_tool_python.LanguageTool('en-GB') 

            spelling_mistakes = len(tool.check(text))

            correct_text = tool.correct(text)

            # Deshacer contracciones
            
            contract = correct_text.count("'")

            correct_text = contractions.fix(correct_text)
            
            tool.close() 
            
            # Obtengo las métricas del texto

            sentences = len(nltk.sent_tokenize(correct_text))

            words = len(nltk.word_tokenize(correct_text))

            words_per_sent = words / sentences

            # Riqueza del lenguaje

            unique_words = len(set(nltk.word_tokenize(correct_text)))

            richness = unique_words / words

            # Numero de palabras que aportan información

            stopwords = nltk.corpus.stopwords.words("english")

            useful_words = list()

            # Elimino los signos de puntuación para analizar el texto

            pattern = r"[^\w\d\s]"

            clean_text = re.sub(pattern, " ", correct_text)

            clean_text = re.sub(f"[ ]+", " ", clean_text)

            for word in nltk.word_tokenize(clean_text):

                if word.casefold() not in stopwords :

                    useful_words.append(word)

            informative = len(useful_words) / words
            
            # Análisis sintáxico / morfológico

            verb = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

            verb_list = list()

            adjective = ["JJ", "JJR", "JJS"]

            adjective_list = list()

            adverb = ["RB", "RBR", "RBS"]

            adverb_list = list()

            blob = TextBlob(correct_text)

            for word in blob.tags:

                    if word[1] in verb:

                        v = Word(word[0]).lemmatize("v")

                        verb_list.append(v)

                    elif word[1] in adjective:

                        adjective_list.append(word[0])

                    elif word[1] in adverb:

                        adverb_list.append(word[0])

            # Tipos de palabras utilizadas

            unique_verbs = len(set(verb_list))

            unique_adjectives = len(set(adjective_list))

            unique_adverbs = len(set(adverb_list))

            uniqueness = int(unique_verbs + unique_adjectives + unique_adverbs)
            
            # Análisis del sentimiento del texto

            blob = TextBlob(correct_text, analyzer = PatternAnalyzer())

            polarity = blob.sentiment[0]

            subjectivity = blob.sentiment[1]

            #blob = TextBlob(text, analyzer = NaiveBayesAnalyzer())

            #positive = blob.sentiment[1]

            #negative = blob.sentiment[2]

            placeholder.empty()

            with placeholder.container():

                col1, col2 = st.columns(2)

                col1.text_area("Texto original:", value = text, height = 700, max_chars = None, key = "original_text", help = "Introduce en el cuadro el texto que desees comprobar", placeholder = "Introduce aquí tu texto")

                col2.text_area("Texto corregido:", value = correct_text, height = 700, max_chars = None, key = "corrected_text", help = "En este cuadro se mostrará el texto corregido", placeholder = "Aquí se mostrarán las correciones", disabled = True)
                
                button = st.button('Corregir texto', key = "correct_button")

                colA, colB, colC, colD, colE = st.columns((1, 2, 1, 2, 1))

                data_1 = pd.DataFrame([["Cohesión", 1.0], ["Sintaxis", 1.5], ["Vocabulario", 2.0]], columns = ["Metric", "Grade"])

                chart_1 = alt.Chart(data_1).mark_bar(cornerRadius = 50, color = "#2A6485").encode(x = alt.X("Grade:Q", axis = None, scale = alt.Scale(domain = [0, 5])), y = alt.Y("Metric:N", title = "", sort = ["Cohesión", "Sintaxis", "Vocabulario"]))

                text = chart_1.mark_text(align = 'left', baseline = 'middle', dx = 3, fontStyle = 'bold', fontSize = 20, color = "#2A6485").encode(text = alt.Text('Grade:Q', format=",.1f"))

                final_bar_1 = (chart_1 + text).properties(height = 200).configure_axis(labelFontSize = 16)

                colB.altair_chart(final_bar_1, use_container_width = True)

                data_2 = pd.DataFrame([["Fraseología", 2.5], ["Gramática", 3.0], ["Convenciones", 3.5]], columns = ["Metric", "Grade"])

                chart_2 = alt.Chart(data_2).mark_bar(cornerRadius = 50, color = "#2A6485").encode(x = alt.X("Grade:Q", axis = None, scale = alt.Scale(domain = [0, 5])), y = alt.Y("Metric:N", title = "", sort = ["Fraseología", "Gramática", "Convenciones"]))

                text = chart_2.mark_text(align = 'left', baseline = 'middle', dx = 3, fontStyle = 'bold', fontSize = 20, color = "#2A6485").encode(text = alt.Text('Grade:Q', format=",.1f"))

                final_bar_2 = (chart_2 + text).properties(height = 200).configure_axis(labelFontSize = 16)

                colD.altair_chart(final_bar_2, use_container_width = True)

                st.markdown('''
                <style>
                /*center metric label*/
                [data-testid="stMetricLabel"] > div:nth-child(1) {
                    justify-content: center;
                }

                /*center metric value*/
                [data-testid="stMetricValue"] > div:nth-child(1) {
                    justify-content: center;
                }
                </style>
                ''', unsafe_allow_html = True)
            
            st.balloons()

            with st.expander("Pincha aquí para ver los resultados de la evaluación :point_down:"):

                col3, col4, col5 = st.columns(3)

                spelling_mistakes_delta = f"{spelling_mistakes - 10} más que la media" if (spelling_mistakes - 10) > 0 else f"{abs(spelling_mistakes - 10)} menos que la media"

                spelling_mistakes_color = "inverse" if (spelling_mistakes - 10) > 0 else "normal"

                col3.metric(label = "Errores gramaticales", value = spelling_mistakes, delta = spelling_mistakes_delta, delta_color = spelling_mistakes_color)

                contract_delta = f"{contract - 10} más que la media" if (contract - 10) > 0 else f"{abs(contract - 10)} menos que la media"

                contract_color = "inverse" if (contract - 10) > 0 else "normal"

                col4.metric(label = "Contracciones", value = contract , delta = contract_delta, delta_color = contract_color)

                uniqueness_delta = f"{uniqueness - 35} más que la media" if uniqueness > 35 else f"{abs(uniqueness - 35)} menos que la media"

                uniqueness_color = "normal" if uniqueness > 35 else "inverse"
                
                col5.metric(label = "Palabras únicas", value = uniqueness, delta = uniqueness_delta, delta_color = uniqueness_color) 

                st.write(" ")

                col6, col7, col8, col9 = st.columns(4)

                col6.metric(label = "Riqueza del lenguaje", value = round(richness, 2), help = "La riqueza del lenguaje es una medición de la cantidad de palabras diferentes frente al total de palabras del texto.")

                col7.metric(label = "Información aportada", value = round(informative, 2), help = "La información aportada es una medición de la cantidad de palabras que aportan información relevante frente al total de palabras del texto.")
                
                col8.metric(label = "Polaridad", value = round(polarity, 2), help = "La polaridad mide la fuerza de las opiniones que aparecen en el texto. Será positiva cuando el sentimiento asociado al texto es una emoción positiva, y negativa en caso contrario.")
                
                col9.metric(label = "Subjectividad", value = round(subjectivity, 2), help = "La subjetividad mide el grado de implicación personal en el texto. Los textos con una subjetividad alta suelen referirse a opiniones personales, emociones o juicios, mientras que los objetivos se refieren a información basada en hechos reales.")

                st.markdown('''
                <style>
                /*center metric label*/
                [data-testid="stMetricLabel"] > div:nth-child(1) {
                    justify-content: center;
                }

                /*center metric value*/
                [data-testid="stMetricValue"] > div:nth-child(1) {
                    justify-content: center;
                }

                /*center metric delta value*/
                div[data-testid="metric-container"] > div[data-testid="stMetricDelta"] > div{
                justify-content: center;
                }

                /*center metric delta svg*/
                [data-testid="stMetricDelta"] > svg {
                position: absolute;
                left: 30%;
                -webkit-transform: translateX(-50%);
                -ms-transform: translateX(-50%);
                transform: translateX(-50%);
                }
                </style>
                ''', unsafe_allow_html = True)

else:

    pass
