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

from utils import * 

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    
    st.title("Binary Classification Web App")
    st.sidebar.title("Control Center")
    st.markdown("Are your mushrooms edible or poisonous? 🍄🍄🍄 ")
    st.sidebar.markdown("To adjust modelling and its parameter")

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']  
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", 
                                                    "Logistic Regression",
                                                    "Random Forest"))

    if classifier == 'Support Vector Machine (SVM)':
        svm_function(x_train, x_test, y_train, y_test, class_names, classifier)
    elif classifier == 'Logistic Regression':
        lr_function(x_train, x_test, y_train, y_test, class_names, classifier)
    elif classifier == 'Random Forest':
        rf_function(x_train, x_test, y_train, y_test, class_names, classifier)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Raw Dataset (Classification)")
        st.write(df)   
    
if __name__ == '__main__':
    main()
