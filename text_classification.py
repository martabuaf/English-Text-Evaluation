## Librerias

# Procesamiento de datos
import numpy as np
import pandas as pd

# Modelo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import MultiOutputClassifier

# Regresion
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost.sklearn import XGBRegressor

# Clasificacion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Metricas
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Visualizaciones
import matplotlib.pyplot as plt
import seaborn as sns


## Carga y procesamiento de datos

df = pd.read_csv("scored_text.csv")

labels = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

for label in labels:

    df[label] = df[label].apply(lambda x: int((x*2)-2.0))

## Parametros de entrenamiento

x = np.array(df[["spelling_mistakes", "contractions", "words_per_sent", "richness", "informative", "unique_verbs", "unique_adjectives", "unique_adverbs", "polarity", "subjectivity"]])

y = np.array(df[["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]])

x_norm = MinMaxScaler().fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size = 0.20, random_state = 42)

## Funciones de comprobacion

def desviacion():
    
    # Muestro la diferencia entre los valores reales y las predicciones

    df_pred = pd.DataFrame()

    df_pred["y_test"] = y_test.flatten()

    df_pred["y_pred"] = y_pred.flatten()

    df_pred["desviación"] = round(abs((df_pred["y_test"] - df_pred["y_pred"] ) ), 2)

    desviacion_media = df_pred["desviación"].mean()

    elementos_desviacion = df_pred[df_pred["desviación"] > 1].shape[0]

    return print(f" Desviación media: {desviacion_media} \t Elementos: {elementos_desviacion}\n")

def metricas():
    
    # Metricas de errores

    # Relative Absolute Error

    RAE = np.sum(np.abs(np.subtract(y_test, y_pred))) / np.sum(np.abs(np.subtract(y_test, np.mean(y_test))))

    # Relative Square Error

    RSE = np.sum(np.square(np.subtract(y_test, y_pred))) / np.sum(np.square(np.subtract(y_test, np.mean(y_test))))

    # Adjusted R**2

    r2_ajustada = 1 - (1 - model.score(x_test, y_test))*(len(y_test) - 1)/(len(y_test) - x_test.shape[1] - 1)

    #  Root Mean Square Error

    score = np.sqrt(mean_squared_error(y_pred, y_test))

    print(f"MAE:\t {mean_absolute_error(y_pred, y_test)}")
    print(f"MSE:\t {mean_squared_error(y_pred, y_test)}")
    print(f"R**2:\t {r2_score(y_pred, y_test)}")
    print(f"RAE:\t {RAE}")
    print(f"RSE:\t {RSE}")
    print(f"Adjusted R^2:\t {r2_ajustada}")
    print(f"RMSE Score: {score}")
    
    return

def grafico():
    
    # Represento las desviaciones en una gráfica

    plt.figure(figsize = (8, 5))

    sns.scatterplot(x = y_test.flatten(), y = y_pred.flatten(), alpha = 0.5, color = "blue")

    plt.xlabel("Valores Reales (y_test)", size = 18)

    plt.ylabel("Predicciones (y_pred)", size = 18)

    return plt.show()

## Metodos de Regresion

regresores = [LinearRegression(), GradientBoostingRegressor(), ElasticNet(), SGDRegressor(), SVR(), BayesianRidge(), KernelRidge()]

for regresor in regresores:

    model = MultiOutputRegressor(regresor).fit(x_train, y_train)

    y_pred = model.predict(x_test)
    
    y_pred = np.around(y_pred) # La regresion devuelve valores continuos así que los convierto a discretos

    # Muestro la diferencia entre los valores reales y las predicciones
    
    print(f"{regresor}:\n")

    desviacion()

# Linear Regression

model = MultiOutputRegressor(LinearRegression(fit_intercept = True, n_jobs = None, positive = False)).fit(x_train, y_train)

y_pred = model.predict(x_test)

y_pred = np.around(y_pred)

desviacion()

# Gradient Boosting Regressor

model = MultiOutputRegressor(GradientBoostingRegressor(
                                                        loss = 'squared_error',
                                                        learning_rate = 0.1,
                                                        n_estimators = 100,
                                                        subsample = 1.0,
                                                        criterion = 'friedman_mse',
                                                        min_samples_split = 2,
                                                        min_samples_leaf = 1,
                                                        min_weight_fraction_leaf = 0.0,
                                                        max_depth = 3,
                                                        min_impurity_decrease = 0.0,
                                                        init = None,
                                                        random_state = None,
                                                        max_features = None,
                                                        alpha = 0.9,
                                                        verbose = 0,
                                                        max_leaf_nodes = None,
                                                        warm_start = False,
                                                        validation_fraction = 0.1,
                                                        n_iter_no_change = None,
                                                        tol = 0.0001,
                                                        ccp_alpha = 0.0)).fit(x_train, y_train)

y_pred = model.predict(x_test)

y_pred = np.around(y_pred)

desviacion()

# Support Vector Regressor

model = MultiOutputRegressor(SVR(
                                    kernel = 'rbf',
                                    degree = 3,
                                    gamma = 'scale',
                                    coef0 = 0.0,
                                    tol = 0.001,
                                    C = 1.0,
                                    epsilon = 0.1,
                                    shrinking = True,
                                    cache_size = 200,
                                    verbose = False,
                                    max_iter = -1)).fit(x_train, y_train)

y_pred = model.predict(x_test)

y_pred = np.around(y_pred)

desviacion()

# Bayesian Ridge

model = MultiOutputRegressor(BayesianRidge(
                                            n_iter = 300,
                                            tol = 0.001,
                                            alpha_1 = 1e-06,
                                            alpha_2 = 1e-06,
                                            lambda_1 = 1e-06,
                                            lambda_2 = 1e-06,
                                            alpha_init = None,
                                            lambda_init = None,
                                            compute_score = False,
                                            fit_intercept = True)).fit(x_train, y_train)

y_pred = model.predict(x_test)

y_pred = np.around(y_pred)

desviacion()

# XGB Regressor

model = MultiOutputRegressor(XGBRegressor(
                                            n_estimators = 100,
                                            max_depth = 1,
                                            max_leaves = 10,
                                            grow_policy = "lossguide",
                                            min_child_weight = 0.1,
                                            max_delta_step = 3,
                                            n_jobs = -1)).fit(x_train, y_train)

y_pred = model.predict(x_test)

y_pred = np.around(y_pred)

desviacion()

## Metodos de Clasificacion

clasificadores = [LogisticRegression(max_iter=1000), KNeighborsClassifier(), NearestCentroid(), GaussianNB(), DecisionTreeClassifier(), RandomForestClassifier(), SVC(), AdaBoostClassifier(), GradientBoostingClassifier()]

for clasificador in clasificadores:

    model = MultiOutputClassifier(clasificador).fit(x_train, y_train)

    y_pred = model.predict(x_test)

    # Muestro la diferencia entre los valores reales y las predicciones
    
    print(f"{clasificador}:\n")

    desviacion()

# Logistic Regression

model = MultiOutputClassifier(LogisticRegression(
                                                    penalty = 'l2',
                                                    dual = False,
                                                    tol = 0.0001,
                                                    C = 1.0,
                                                    fit_intercept = True,
                                                    intercept_scaling = 1,
                                                    class_weight = None,
                                                    random_state = None,
                                                    solver = 'lbfgs',
                                                    max_iter = 500,
                                                    multi_class = 'auto',
                                                    verbose = 0,
                                                    warm_start = False,
                                                    n_jobs = -1,
                                                    l1_ratio = None)).fit(x_train, y_train)

y_pred = model.predict(x_test)

desviacion()

# Random Forest Classifier

model = MultiOutputClassifier(RandomForestClassifier(
                                                        n_estimators = 1000,
                                                        criterion = 'gini',
                                                        max_depth = 8,
                                                        min_samples_split = 2,
                                                        min_samples_leaf = 1,
                                                        min_weight_fraction_leaf = 0.0,
                                                        max_features = 'sqrt',
                                                        max_leaf_nodes = None,
                                                        min_impurity_decrease = 0.0,
                                                        bootstrap = True,
                                                        oob_score = False,
                                                        n_jobs = None,
                                                        random_state = None,
                                                        warm_start = False,
                                                        class_weight = None,
                                                        ccp_alpha = 0.0,
                                                        max_samples = None)).fit(x_train, y_train)

y_pred = model.predict(x_test)

desviacion()

# Support Vector Classifier

model = MultiOutputClassifier(SVC(
                                    kernel = 'rbf',
                                    degree = 3,
                                    gamma = 'scale',
                                    coef0 = 0.0,
                                    tol = 0.001,
                                    C = 8.0,
                                    shrinking = True,
                                    cache_size = 200,
                                    max_iter = -1)).fit(x_train, y_train)

y_pred = model.predict(x_test)

desviacion()

# Gradient Boosting Classifier

model = MultiOutputClassifier(GradientBoostingClassifier(
                                                        loss = 'log_loss',
                                                        learning_rate = 0.1,
                                                        n_estimators = 100,
                                                        subsample = 1.0,
                                                        criterion = 'friedman_mse',
                                                        min_samples_split = 2,
                                                        min_samples_leaf = 1,
                                                        min_weight_fraction_leaf = 0.0,
                                                        max_depth = 3,
                                                        min_impurity_decrease = 0.0,
                                                        init = None,
                                                        random_state = None,
                                                        max_features = 5,
                                                        verbose = 0,
                                                        max_leaf_nodes = 10,
                                                        warm_start = False,
                                                        validation_fraction = 0.1,
                                                        n_iter_no_change = None,
                                                        tol = 0.0001,
                                                        ccp_alpha = 0.0)).fit(x_train, y_train)

y_pred = model.predict(x_test)

desviacion()
