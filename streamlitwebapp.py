import numpy as np
import pickle
import streamlit as st
import os

model_path = os.path.join(os.getcwd(), 'performance_model.sav')

#load the model
loaded_model= pickle.load(open(model_path,'rb'))



# creating a function for Prediction

def performance_prediction(input_data):
    

     #changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] ==0):
        return 'Low Performance'
    else:
        return 'High Performance'
    

def main():
    
    
    # giving a title
    st.title('Player_Performance Prediction Web App')
    
    
    # getting the input data from the user
    potential= st.text_input('potential')
    value_eur = st.text_input('value_eur')
    wage_eur   = st.text_input('wage_eur')
    age    = st.text_input('age')
    height_cm = st.text_input('height_cm')
    weight_kg = st.text_input('weight_kg')
    league_level= st.text_input('league_level')
    club_jersey_number  = st.text_input('club_jersey_number')
    club_contract_valid_until= st.text_input('club_contract_valid_until')
    weak_foot      = st.text_input('weak_foot')
    skill_moves = st.text_input('skill_moves')
    release_clause_eur = st.text_input('release_clause_eur')
    pace      = st.text_input('pace')
    shooting = st.text_input('shooting')
    passing  = st.text_input('passing')
    dribbling = st.text_input('dribbling')
    defending    = st.text_input('defending')
    physic   = st.text_input('physic')
    attacking_crossing = st.text_input('attacking_crossing')
    attacking_finishing  = st.text_input('attacking_finishing')
    attacking_heading_accuracy = st.text_input('attacking_heading_accuracy')
    preferred_foot_Right = st.text_input('preferred_foot_Right')
    work_rate_High_Low   = st.text_input('work_rate_High_Low')
    work_rate_High_Medium= st.text_input('work_rate_High_Medium')
    work_rate_Low_High = st.text_input('work_rate_Low_High')
    work_rate_Low_Low = st.text_input('work_rate_Low_Low')
    work_rate_Low_Medium = st.text_input('work_rate_Low_Medium')
    work_rate_Medium_High  = st.text_input('work_rate_Medium_High')
    work_rate_Medium_Low = st.text_input('work_rate_Medium_Low')
    work_rate_Medium_Medium = st.text_input('work_rate_Medium_Medium')
    
    
    
   
    
    
    # code for Prediction
    performance = ''
    
    # creating a button for Prediction
    
    if st.button('players Test Result'):
        performance = performance_prediction([ work_rate_High_Low,work_rate_High_Medium,
            work_rate_Low_High,work_rate_Low_Low,work_rate_Low_Medium,
work_rate_Medium_High,work_rate_Medium_Low,work_rate_Medium_Medium,
preferred_foot_Right,potential,value_eur,wage_eur,
age,height_cm,weight_kg,league_level,club_jersey_number,club_contract_valid_until,weak_foot,skill_moves,
release_clause_eur,pace,shooting,passing,dribbling,defending,
physic,attacking_crossing,attacking_finishing,attacking_heading_accuracy])
        
    st.success(performance)


if __name__ == '__main__':
    main()
