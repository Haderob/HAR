# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:42:34 2023

@author: Bese
"""

import numpy as np
import pandas as pd
import pickle 
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('C:/Users/Bese/Desktop/ML Project/IEDE project/HAR_trained_model.sav', 'rb'))

#creating a fuction for prediction
def har_prediction(spectra_df):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(spectra_df)
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0] == 0):
        print("The Person is Standing still")

    elif (prediction[0] == 1):
      return ("The person is Sitting still")
    elif (prediction[0] == 2):
      return ("The person isTalking with hand movements while sitting")
    elif (prediction[0] == 3):
      return ("The person is Talking with hand movements while standing or walking")
    elif (prediction[0] == 4):
      return ("The person is Repeatedly standing up and sitting down")
    elif (prediction[0] == 5):
      return ("The person is Laying still")
    elif (prediction[0] == 6):
      return ("The person is Repeatedly standing up and laying down")
    elif (prediction[0] == 7):
      return ("The person is Picking up an object from the floor")
    elif (prediction[0] == 8):
      return ("The person is Jumping repeatedly")
    elif (prediction[0] == 9):
      return ("The person is Performing full push-ups")
    elif (prediction[0] == 10):
      return ("The person is Performing sit-ups")
    elif (prediction[0] == 11):
      return ("The person is Walking 20 meters")
    elif (prediction[0] == 12):
      return ("The person is Walking backward for 20 meters")
    elif (prediction[0] == 13):
      return ("The person is Walking along a circular path")
    elif (prediction[0] == 14):
      return ("The person is Running 20 meters")
    elif (prediction[0] == 15):
      return ("The person is Ascending on a set of stairs ")
    elif (prediction[0] == 16):
      return ("The person is Descending from a set of stairs")
    else:
      return ("The person is Playing table tennis")

  
    
def main():
    # giving a title for the web app
    st.title('Humman Activity Recognition App')
    
    #getting the impute data from the user
    
    spectra = st.file_uploader("upload file", accept_multiple_files=False)
    spectra_df=''
    if spectra is not None:
        spectra_df = pd.read_csv(spectra)
    st.write(spectra_df)
    
    
    # Pregnancies = st.text_input('Number of Pregnacies')
    # Glucose = st.text_input('Glucose Level')
    # BloodPressure = st.text_input('Blood Pressure values')
    # SkinThickness = st.text_input('Skin tickness values')
    # Insulin = st.text_input('Insuline level')
    # BMI = st.text_input('BMI values')
    # DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function values')
    # Age = st.text_input('Age of the person')
    
    
    
    #code for Prediction
    test = ''
    # creating a button for Prediction
    if st.button('HAR Test Result'):
        test=har_prediction([spectra_df])
        
        
        
    st.success(test)





if __name__=='__main__':    
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    