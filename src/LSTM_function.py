import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras import layers, metrics
from keras.models import Sequential
import mlflow
import mlflow.keras

np.random.seed(42)
scaler = MinMaxScaler()
RMSE = metrics.RootMeanSquaredError()

MODEL_CONFIG = {
    "time_steps": 15,
    "lstm_neurons": 300,
    "dense_neurons": 150,
    "structure_name": "LSTM_TS",
    "batch_size": 5,
    "epochs": 400
}


class LSTM():
    def __init__(self, train, test):
        self.set_values(MODEL_CONFIG)
        self.train = train
        self.test = test
        self.rmse_value = None
        self.prediction_values = None
        self.model = None

    def set_values(self, params):
        for key in params:
            setattr(self, key, params[key])
        

    def get_XY(self, dat, time_steps):
        # Prepare Y
        Y_ind = np.arange(time_steps, len(dat), time_steps)
        Y = dat[Y_ind]

        # Prepare X
        rows_x = len(Y)
        X = dat[range(time_steps*rows_x)]
        X = np.reshape(X, (rows_x, time_steps, 1))    
        
        return X, Y


    def scale_data(self, xtrain, xtest):
        X_train = np.array(xtrain).reshape(-1, 1)
        X_test = np.array(xtest).reshape(-1, 1)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test).reshape(-1)
        return X_train_scaled, X_test_scaled


    def inverse_vec(self, vector):
        return scaler.inverse_transform(vector).flatten()

    
    def plot_predicted(self):

        if self.prediction_values is None:
            raise ValueError("Run the model first using the 'run()' function.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.array(range(0,len(self.test))),
                                y=self.test,
                                mode='lines',
                                name='Line 1',
                                opacity=0.8,
                                line=dict(color='black', width=1)
                                ))
        fig.add_trace(go.Scatter(x=np.array(range(0,len(self.prediction_values))),
                                y=self.prediction_values,
                                mode='lines',
                                name='Line 2',
                                opacity=0.8,
                                line=dict(color='red', width=1)
                                ))

        fig.update_layout(dict(plot_bgcolor = 'white'))

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                        zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                        showline=True, linewidth=1, linecolor='black',
                        title='Observation'
                        )

        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                        zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                        showline=True, linewidth=1, linecolor='black',
                        title='Title'
                        )

        fig.update_layout(title=dict(text="Title", 
                                    font=dict(color='black')),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )

        fig.show()


    def init_model(self):
        self.model = Sequential( 
            [
                layers.LSTM(self.lstm_neurons, activation='tanh', input_shape=(self.time_steps, 1)),
                layers.Dense(self.dense_neurons, activation = "linear"),
                layers.Dense(units = 1, activation = "linear"),
            ]
        )


    def run(self):
        '''Runs the LSTM network.
           If the model was loaded previously, this function will use that model for the forecasting and 
           plots. Othervise it will train a new model.'''
           
        # ---------- Data prep
        X_train_scaled, X_test_scaled = self.scale_data(self.train, self.test)
        trainX, trainY = self.get_XY(X_train_scaled, self.time_steps)

        if self.model is None:
            self.train_model(trainX, trainY)

        x_test = X_train_scaled[-self.time_steps:]
        for i in range(len(X_test_scaled)):
            predicted = self.model.predict(x_test[-self.time_steps:].reshape(-1, self.time_steps, 1))
            x_test = np.concatenate((x_test, predicted))

        # ---------- Convert normalized values back to normal
        true_normal = self.inverse_vec(X_test_scaled.reshape(-1, 1))
        predicted_normal = self.inverse_vec(x_test[-len(X_test_scaled):])

        # ---------- Calc RMSE
        predicted_rmse = RMSE(true_normal, predicted_normal).numpy()

        # ---------- Set predicted values
        self.rmse_value = predicted_rmse
        self.prediction_values = predicted_normal


    def train_model(self, trainX, trainY):
        '''Run the LSTM network'''
        self.init_model()
        self.model.compile(loss='mean_squared_error', 
                           metrics=['RootMeanSquaredError'], 
                           optimizer='adam')


        with mlflow.start_run() as run:
            # ---------- Logging to MLFlow
            mlflow.log_param("time_steps", self.time_steps)
            mlflow.log_param("neurons", self.lstm_neurons)
            mlflow.log_param("structure_name", self.structure_name)
            mlflow.tensorflow.autolog()

            self.model.fit(trainX,
                           trainY,
                           batch_size=self.batch_size,
                           epochs=self.epochs,
                           shuffle=True,
                           )


    def save_model(self, model_name = "models/model.h5"):
        '''Saves the model into the "model_name" file'''
        self.model.save_weights(model_name)

    
    def load_model(self, model_name = "models/model.h5"):
        '''Loads the model from the "model_name" file.
           The structure of the model file must match the structure initialized with the "init_model" 
           function'''
        try:
            self.init_model(self.lstm_neurons)
            self.model.load_weights(model_name)
        except:
            raise ValueError("The weights structure does not match the model structure.")



    def rmse(self):
        '''Returns the Root mean square error (RMSE) value'''
        if self.rmse_value is None:
            raise ValueError("Run the model first using the 'run()' function.")
        return self.rmse_value


    def predictions(self):
        '''Returns the predicted time series. The samples of prediction matches the length of the test time 
           series'''
        if self.prediction_values is None:
            raise ValueError("Run the model first using the 'run()' function.")
        return self.prediction_values

    
    def plot_predictions(self):
        '''Plot the tested VS predicted values'''
        self.plot_predicted()