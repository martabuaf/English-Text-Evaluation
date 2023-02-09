# Procesamiento de datos
import numpy as np
import pandas as pd

# Herramientas de gramática
import language_tool_python
import contractions
import re

# Procesamiento de lenguaje natural
import nltk
from textblob import TextBlob, Word
from textblob.sentiments import PatternAnalyzer, NaiveBayesAnalyzer

# Integración
import streamlit as st
import base64

# Introduzco el texto

st.set_page_config(page_title = "Evalúa tu inglés ahora!", page_icon = "/Users/marta/DS NOTEBOOK/Modulo 0_Proyectos/English Text Evaluation/page-icon.png", layout = "wide", initial_sidebar_state = "expanded")

st.markdown("<h1 style='text-align: center; font-family: Verdana;color: white;'>Evalúa al instante tu nivel de inglés</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: white;'>Con este asistente de escritura gratuito, compruebe si hay errores gramaticales, de estilo y ortográficos en su texto en inglés</h1>", unsafe_allow_html=True)

placeholder = st.empty()

with placeholder.container():

    col1, col2 = st.columns(2)

    text = col1.text_area("Texto original:", height = 700, max_chars = None, key = None, help = "Introduce en cuadro el texto que desees comprobar", placeholder = "Introduce aquí tu texto")

    col2.text_area("Texto corregido:", height = 700, max_chars = None, key = None, help = "En este cuadro se mostrará el texto corregido", placeholder = "Aquí se mostrarán las correciones", disabled = True)

    button = st.button('Corregir texto')

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

set_background('/Users/marta/DS NOTEBOOK/Modulo 0_Proyectos/English Text Evaluation/background-file.png')

# Busco y corrijo los errores del texto

def check_mistakes(text, tool = language_tool_python.LanguageTool('en-GB')): 

    # Limpieza de formato

    #pattern = r"[^\w\.',]"

    #text = re.sub(pattern, " ", text)

    #text = re.sub(f"[ ]+", " ", text)

    # Errores ortográficos y tipográficos    

    spelling_mistakes = len(tool.check(text))

    text = tool.correct(text)

    # Deshacer contracciones
    
    contract = text.count("'")

    correct_text = contractions.fix(text)
    
    tool.close() 

    return spelling_mistakes, contract, correct_text
    
# Obtengo las métricas del texto
    
def get_metrics(text): 

    # Numero de palabras por oracion

    sentences = len(nltk.sent_tokenize(text))

    words = len(nltk.word_tokenize(text))

    words_per_sent = words / sentences

    # Riqueza del lenguaje

    unique_words = len(set(nltk.word_tokenize(text)))

    richness = unique_words / words

    # Numero de palabras que aportan información

    stopwords = nltk.corpus.stopwords.words("english")

    useful_words = list()

    # Elimino los signos de puntuación para analizar el texto

    pattern = r"[^\w\d\s]"

    clean_text = re.sub(pattern, " ", text)

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

    blob = TextBlob(text)

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
    
    # Análisis del sentimiento del texto

    blob = TextBlob(text, analyzer = PatternAnalyzer())

    polarity = blob.sentiment[0]

    subjectivity = blob.sentiment[1]
    
    # Análisis del sentimiento del texto

    #blob = TextBlob(text, analyzer = NaiveBayesAnalyzer())

    #positive = blob.sentiment[1]

    #negative = blob.sentiment[2]
    
    return words_per_sent, richness, informative, unique_verbs, unique_adjectives, unique_adverbs, polarity, subjectivity
    
    # Streamlit

    
if button == True:
    
    if text == "":
    
        st.warning('Introduce un texto de ejemplo')

    else:
    
        spelling_mistakes, contract, correct_text = check_mistakes(text)
        
        words_per_sent, richness, informative, unique_verbs, unique_adjectives, unique_adverbs, polarity, subjectivity = get_metrics(correct_text)

        placeholder.empty()

        with placeholder.container():

            col1, col2 = st.columns(2)

            col1.text_area("Texto original:", value = text, height = 700, max_chars = None, key = 738, help = "Introduce en el cuadro el texto que desees comprobar", placeholder = "Introduce aquí tu texto")

            col2.text_area("Texto corregido:", value = correct_text, height = 700, max_chars = None, key = 739, help = "En este cuadro se mostrará el texto corregido", placeholder = "Aquí se mostrarán las correciones", disabled = True)
            
            button = st.button('Corregir texto', key = 740)

        st.balloons()

else:

    pass
