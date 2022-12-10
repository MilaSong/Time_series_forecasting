import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras import layers, metrics
from keras.models import Sequential
import mlflow
import keras
import mlflow.keras

np.random.seed(42)
scaler = MinMaxScaler()
rmse = metrics.RootMeanSquaredError()


def get_XY(dat, time_steps):
    # Prepare Y
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    Y = dat[Y_ind]

    # Prepare X
    rows_x = len(Y)
    X = dat[range(time_steps*rows_x)]
    X = np.reshape(X, (rows_x, time_steps, 1))    
    
    return X, Y


def train_test_split(split_by = "YYYY-mm-dd"):
    return df[df.DATE < split_by], df[df.DATE >= split_by]


def scale_data(xtrain, xtest):
    X_train = np.array(xtrain["count"]).reshape(-1, 1)
    X_test = np.array(xtest["count"]).reshape(-1, 1)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test).reshape(-1)
    return X_train_scaled, X_test_scaled


def create_model(neurons=50):
    model = Sequential( 
        [
            layers.LSTM(neurons, activation='tanh', input_shape=(time_steps, 1)),
            layers.Dense(units = 150, activation = "linear"),
            layers.Dense(units = 1, activation = "linear"),
        ]
    )
    return model


def inverse_vec(vector):
    return scaler.inverse_transform(vector).flatten()


def plot_predicted(true_data, pred_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(range(0,len(true_data))),
                            y=true_data,
                            mode='lines',
                            name='Line 1',
                            opacity=0.8,
                            line=dict(color='black', width=1)
                            ))
    fig.add_trace(go.Scatter(x=np.array(range(0,len(pred_data))),
                            y=pred_data,
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


# ---------- Constants
time_steps = 50
neurons_number = 300
STRUCTURE_NAME = "ManyToOne2"
structure_index = 12


# ---------- Data prep
df = pd.read_csv("data/preprocessed/clean_data_days.csv")
train_data, test_data = train_test_split()
X_train_scaled, X_test_scaled = scale_data(train_data, test_data)
trainX, trainY = get_XY(X_train_scaled, time_steps)

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model = create_model(neurons_number)
model.compile(loss='mean_squared_error', 
              metrics=['RootMeanSquaredError'], 
              optimizer='adam')


with mlflow.start_run() as run:

    # ---------- Logging to MLFlow
    mlflow.log_param("time_steps", time_steps)
    mlflow.log_param("neurons", neurons_number)
    mlflow.log_param("structure_name", STRUCTURE_NAME)
    mlflow.log_param("structure_index", structure_index)
    mlflow.tensorflow.autolog()

    history = model.fit(trainX,
                        trainY,
                        batch_size=5,
                        epochs=200,
                        callbacks=[callback]
                        )


    # ---------- Prediction
    x_test = X_train_scaled[-time_steps:]
    for i in range(len(test_data)):
        predicted = model.predict(x_test[-time_steps:].reshape(-1, time_steps, 1))
        x_test = np.concatenate((x_test, predicted))

    # ---------- Convert normalized values back to normal
    true_normal = inverse_vec(X_test_scaled.reshape(-1, 1))
    predicted_normal = inverse_vec(x_test[-len(X_test_scaled):])

    # ---------- Calc and log RMSE
    predicted_rmse = rmse(true_normal, predicted_normal).numpy()
    mlflow.log_metric("predicted_rmse", predicted_rmse)

    # ---------- Plot predicted test set
    plot_predicted(true_normal, predicted_normal)
