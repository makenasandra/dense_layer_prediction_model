
# Predicting Total Video Game Earnings using Dense Layer Neural Network
In this project the dataset contains video games sold by a retailer and earnings for each video game
<br>We will use the data to train the Neural Network(NN) to predict how much we expect future video games to earn
<br>Each column represents a game attribute 
<br>Each row represents a game sold in the past
<br>We look at each characteristic of the game to predict expected earnings
<br>We will use Kera for the the front end layer and backend will be powered by Theano
<br>Let's begin!

### Preprocess Data
We first start by scaling our data using the MinMaxSCaler from the SkLearn library


```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load training data set from CSV file
training_data_df = pd.read_csv("sales_data_training.csv")

# Load testing data set from CSV file
test_data_df = pd.read_csv("sales_data_test.csv")

# Data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well.
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
#Fit_transfrom allows us to first fit the scaler to our data... 
#...that is figure out how much to scale down the values for each feature(in each clomun)...
#...then it will scale(transform) the data
scaled_training = scaler.fit_transform(training_data_df)
#Using transform instead of fit_transfrom allows the scaler to scale down the test data in the same measure as the train data
scaled_testing = scaler.transform(test_data_df) 

# Print out the adjustment that the scaler applied to the total_earnings column of data
print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))

# Create new pandas DataFrame objects from the scaled data(which is are now plain arrays)
#This allows us to take advantage of pandas capability to save CSV files which only works for DF objects
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)

# Save scaled data dataframes to new CSV files
scaled_training_df.to_csv("sales_data_training_scaled.csv", index=False)
scaled_testing_df.to_csv("sales_data_testing_scaled.csv", index=False)
```

    C:\Users\Sandra\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:323: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by MinMaxScaler.
      return self.partial_fit(X, y)
    

    Note: total_earnings values were scaled by multiplying by 0.0000036968 and adding -0.115913
    

## Create Model
We will use the **Keras Sequential Model** which one of the easiest ways to create a model using Keras


```python
from keras.models import Sequential
from keras.layers import *

#We load the prescaled training data from the newly saved CSV files
training_data_df = pd.read_csv("sales_data_training_scaled.csv")

#We split the data into two the input values and expected outputs
#For X we use all the columns and drop the total_earnings
X = training_data_df.drop('total_earnings', axis=1).values

#Y only Contains expected earnings for each game
Y = training_data_df[['total_earnings']].values

# Define the model using Keras Sequential API model
#Create Sequential object
odel = Sequential()
#Create layers
#Input layer with 50 densely connected nodes, input size(no.of features) and ReLU activation function
model.add(Dense(50, input_dim=9, activation='relu'))

#We can add or remove layers as we experiment
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))

#Our expected output(total earnings) is a single linear value... 
#...thus we have one node in dense layer and use linear activation function(which is the default)
model.add(Dense(1, activation='linear'))

#We use mean_squared_error(mse) loss function to evaluate accuracy of predictions against the actual 
model.compile(loss="mse", optimizer="adam")
```

## Train Model
**Epoch** - is a single training pass(iteration) across the entire dataset /n
<br>**Verobe** - shuffles order of the training data examples randomly as NN train best when data is shuffled


```python
#import pandas as pd
#from keras.models import Sequential
#from keras.layers import *

# Train the model using fit function
model.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    verbose=2
)

# Load the separate test data set  to determine if the NN has 'learned' 
test_data_df = pd.read_csv("sales_data_test_scaled.csv")

#Separe input and output columns for test data
X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df[['total_earnings']].values

#We use model.evaluate to measure the accuracy of the trained model with the test data set as measured by cost function(mse)
test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
```

## Make predictions with new data


```python
# Load the data we want to use to make a prediction
#The data is already preprocessed
X = pd.read_csv("proposed_new_product.csv").values

# Make a prediction with the neural network
prediction = model.predict(X)

#Keras usually assumes that we are going to ask for mutiple predictions with multiple output values... 
#...thus it always returns predictions as a 2D array
# In our case we extract the first element of the first prediction (since that's the only one we have)
prediction = prediction[0][0]

# Re-scale the data from the 0-to-1 range(scaled form) back to dollars
# These constants are from when the data was originally scaled down to the 0-to-1 range
prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

print("Earnings Prediction for Proposed Product - ${}".format(prediction))
```

## Save the trained model
We train the model once and save it to a file so that when we want to use later we can just load it and use it 
<br>**hf file extension**- is a binary format designed for storing python array data


```python
# Save the model to disk in hdf5 format
#Both the structure and trained weights(that determine how the model works) are saved
model.save("trained_model.h5")
print("Model saved to disk.")

```

### Load saved model (optional)
Once the new file is created and our model is saved in it, we can load the trained neural network from another file using the steps below:


```python
from keras.models import load_model

#Loading the model recreated the our entire trained NN
model = load_model('trained_model.h5')

# Load the data we want to use to make a prediction for
X = pd.read_csv("proposed_new_product.csv").values
prediction = model.predict(X)

# Grab just the first element of the first prediction (since we only have one)
prediction = prediction[0][0]

# Re-scale the data from the 0-to-1 range back to dollars
# These constants are from when the data was originally scaled down to the 0-to-1 range
prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

print("Earnings Prediction for Proposed Product - ${}".format(prediction))
```

