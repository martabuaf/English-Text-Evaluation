## Librerías

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

# Visualizaciones
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

## Cargo el dataset

df = pd.read_csv("text_data.csv")

# Dividimos en train y test

df_train = df[0:3130] # 80% de los datos

df_test = df[3130:] # 20% de los datos

## Paso 1: Procesamiento de los datos

def check_mistakes(text, tool = language_tool_python.LanguageTool('en-GB')): # Busco y corrijo los errores de un solo texto

    # Limpieza de formato

    pattern = r"[^\w\.',]"

    text = re.sub(pattern, " ", text)

    text = re.sub(f"[ ]+", " ", text)

    # Errores ortográficos y tipográficos    

    spelling_mistakes = len(tool.check(text))

    text = tool.correct(text)

    # Deshacer contracciones
    
    contract = text.count("'")

    correct_text = contractions.fix(text)

    return spelling_mistakes, contract, correct_text

def check_data(df):  # Busco y corrijo los errores de todos los textos del DataFrame
    
    # Inicializo las listas
    
    spelling_mistakes_list = list()
    
    contract_list = list()
    
    correct_text_list = list()
    
    # Inica el servidor
    
    tool = language_tool_python.LanguageTool('en-GB')  # Servidor local

    # Analizo los textos

    for i in range(len(df)):

        text = df["full_text"][i]

        spelling_mistakes, contract, correct_text = check_mistakes(text, tool)
        
        spelling_mistakes_list.append(spelling_mistakes)
        
        contract_list.append(contract)
        
        correct_text_list.append(correct_text)
    
    # Cierra el servidor
    
    tool.close() 

    # Añado los valores al DataFrame

    df["spelling_mistakes"] = np.array(spelling_mistakes_list)
    
    df["contractions"] = np.array(contract_list)

    df["correct_text"] = np.array(correct_text_list)
    
    return df

# Ejecuto el procesamiento de los datos del dataframe

correct_df = check_data(df_train)

correct_df.to_csv("corrected_text.csv", index = False)

## Paso 2: Procesamiento del lenguaje natural (NLP)

correct_df = pd.read_csv("corrected_text.csv")

def get_metrics(text): # Obtengo las métricas de un solo texto

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

def get_metrics_data(df):  # Obtengo las métricas de todos los textos del DataFrame
    
    # Inicializo las listas
    
    words_per_sent_list = list()
    
    richness_list = list()
    
    informative_list = list()
    
    unique_verbs_list = list()
    
    unique_adjectives_list = list()
    
    unique_adverbs_list = list()
    
    polarity_list = list()
    
    subjectivity_list = list()
    
    spelling_mistakes_list = list()
    
    contract_list = list()
    
    correct_text_list = list()

    # Analizo los textos

    for i in range(len(df)):

        text = df["correct_text"][i]

        words_per_sent, richness, informative, unique_verbs, unique_adjectives, unique_adverbs, polarity, subjectivity = get_metrics(text)
        
        words_per_sent_list.append(words_per_sent)
    
        richness_list.append(richness)
        
        informative_list.append(informative)

        unique_verbs_list.append(unique_verbs)

        unique_adjectives_list.append(unique_adjectives)

        unique_adverbs_list.append(unique_adverbs)

        polarity_list.append(polarity)
        
        subjectivity_list.append(subjectivity)

    # Añado los valores al DataFrame

    df["words_per_sent"] = np.array(words_per_sent_list)
    
    df["richness"] = np.array(richness_list)
        
    df["informative"] = np.array(informative_list)

    df["unique_verbs"] = np.array(unique_verbs_list)

    df["unique_adjectives"] = np.array(unique_adjectives_list)

    df["unique_adverbs"] = np.array(unique_adverbs_list)

    df["polarity"] = np.array(polarity_list)
    
    df["subjectivity"] = np.array(subjectivity_list)
    
    return df

# Ejecuto el procesamiento de los datos del dataframe

scored_df = get_metrics_data(correct_df)

scored_df.to_csv("scored_text.csv", index = False)

# Matriz de correlaciones

plt.figure(figsize=(20,20))  

sns.heatmap(scored_df.corr(), annot = True, cmap = "coolwarm", center = 0)

plt.title(label = "Matriz de correlaciones")

plt.show()

## Paso 3: Clasificación

##continuar##

## Paso 4: Evaluación de los resultados

##continuar##
