import numpy as np
import pickle

#load the model
loaded_model= pickle.load(open('C:/Users/512GB/OneDrive/Documents/Major_Files/Projects/Players_Performance/performance_streamlit/performance_model.sav','rb'))


input_data = (73.0,525000.0,22000.0,34.0,180.0,75.0,1.0,16.0,2021.0,3.0,1.0,3600000.0,69.0,54.0,58.0,64.0,56.0,66.0
,12.0,11.0,13.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,1.0
)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] ==0):
  print('Low Performance')
else:
  print('High Performance')
