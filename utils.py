#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:34:05 2024

@author: okguser
"""


import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score

SEED = 30

@st.cache_data(persist=True)
def load_data():
    
    data = pd.read_csv('src/data/mushrooms.csv')
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
        
    return data

@st.cache_data(persist=True)
def split(dataframe):
    y = dataframe['class']
    x = dataframe.drop(columns = ['class'])
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 30, 
                                                        random_state = SEED)
    return x_train, x_test, y_train, y_test 
    
def plot_metrics(metrics_list):
    if 'Confusion Matrix' in metrics_list: 
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels = class_names)
        st.pyplot()
    elif 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        RocCurveDisplay(model, x_test, y_test, display_labels = class_names)
    elif 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        PrecisionRecallDisplay(model, x_test, y_test, display_labels = class_names)

def svm_function(x_train, x_test, y_train, y_test, class_names, classifier):

    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization Parameter) - Range (0,10)", 
                                    0.01, 10.0, step = 0.01, 
                                    key = 'C_SVM')
    kernel = st.sidebar.radio("Kernel", ("RBF", "Linear","Poly","Sigmoid"),
                                  key = 'kernel')
    gamma = st.sidebar.radio("Gamma (Kernal Coefficient)",("Scale","Auto"), 
                                 key = "gamma")
        
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Support Vector Machine (SVM) Results")
        model = SVC(C = C, kernel = kernel.lower(), gamma = gamma.lower())
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(4))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(4))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names)
        st.write("Confusion Matrix: ")
        st.pyplot(plt.show())
        RocCurveDisplay.from_predictions(y_test, y_pred)
        st.write("ROC Curve: ")
        st.pyplot()
        PrecisionRecallDisplay.from_predictions(y_test, y_pred)
        st.write("Precision Recall Curve: ")
        st.pyplot()

def lr_function(x_train, x_test, y_train, y_test, class_names, classifier):

    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization Parameter) - Range (0,10)", 
                                    0.01, 10.0, step = 0.01, 
                                    key = 'C_LR')
    max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Logistic Regression Results")
        model = LogisticRegression(C=C, max_iter = max_iter)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(4))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(4))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names)
        st.write("Confusion Matrix: ")
        st.pyplot(plt.show())
        RocCurveDisplay.from_predictions(y_test, y_pred)
        st.write("ROC Curve: ")
        st.pyplot()
        PrecisionRecallDisplay.from_predictions(y_test, y_pred)
        st.write("Precision Recall Curve: ")
        st.pyplot()

def rf_function(x_train, x_test, y_train, y_test, class_names, classifier): 

    st.sidebar.subheader("Model Hyperparameters")
    n_estimators = st.sidebar.number_input("Number of trees in the forest: ", 50, 5000, step = 10, key = "n_estimators")
    max_depth = st.sidebar.number_input("Maximum depth of the tree: ", 1,20, step = 1, key = "max_depth")
    bootstrap = st.sidebar.radio("Bootstrap samples when building tree: ", ('True', 'False'), key = "bootstrap")

    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Random Forest Results")
        model = RandomForestClassifier(n_estimators = n_estimators,max_depth = max_depth,bootstrap = bool(bootstrap), n_jobs = -1)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(4))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(4))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names)
        st.write("Confusion Matrix: ")
        st.pyplot(plt.show())
        RocCurveDisplay.from_predictions(y_test, y_pred)
        st.write("ROC Curve: ")
        st.pyplot()
        PrecisionRecallDisplay.from_predictions(y_test, y_pred)
        st.write("Precision Recall Curve: ")
        st.pyplot()







