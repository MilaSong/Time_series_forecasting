# Time series homework

Scripts for the time series subject P160M102 Homework task. The task was to forecast time series using general time series forecasting methods and create one hybrid model. For the hybrid model, LSTM neural network was selected. This network was combined togehter with ARIMA family models in order to forecast different combinations of time series components.

The main code is in the `src` folder:
- `LSTM_function.py` is the main function of the LSTM model loaded in the R files.
- `LSTM_testing.py` is a python script to test the LSTM capabilities of time series forecasting.
- `ts_analysis.R` has the R scripts to analyze the time series data.
- `ts_forecasting.R` has the R scripts to forecast time series using general methods.
- `hybrid_model.R` has the R scripts of the hybrid model. This file loads the LSTM function file.
- `plot_data.py` is a helper function to plot the time series data.

The data preprocessing scripts are in the `utils` folder.


