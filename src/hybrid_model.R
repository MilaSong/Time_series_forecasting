library(reticulate)
library(ggplot2)
library(forecast)
library(tseries)
source_python('utils/split_data.py')
source_python('src/LSTM_function.py')


calc.rmse <- function(predictions) {
    sqrt(mean((as.numeric(testing.ts) - as.numeric(predictions))^2))
}


# ------------- Get T. S.
df <- split_by_days()
data.ts <- ts(df$count, start=YYYY, frequency=365.25)
clean.ts <- tsclean(data.ts)


# ------------- Train/Test split
# Train set: first three years 
train.ts <- ts(clean.ts[1:1096], start=YYYY, frequency=365.25)

# Test set: remaining YYYY+3 years
testing.ts <- ts(clean.ts[1097:length(clean.ts)], start=YYYY+3, frequency=365.25)


# ------------- Decomposing the T. S.
# Simple decomposition of T.S
decomposed <- stl(train.ts, s.window=7)


# ------------- ARMA model forecasting
#model <- auto.arima(decomposed$time.series[,1] + decomposed$time.series[,3], trace=FALSE, seasonal=TRUE, stationary=TRUE, stepwise=FALSE, max.order=10, parallel=TRUE, num.cores=30)
model <- auto.arima(decomposed$time.series[,3], trace=FALSE, seasonal=TRUE, stationary=TRUE, stepwise=FALSE, max.order=10, parallel=TRUE, num.cores=30)
summary(model)

fc4 <- forecast(model, h=length(testing.ts))$mean
plot(testing.ts)
lines(fc4, col=2)

rmse <- calc.rmse(fc4)
print(rmse)


# ------------- LSTM forecasting
lstm <- LSTM(decomposed$time.series[,2] + decomposed$time.series[,1], testing.ts)
#lstm$load_model(model_name="season_trend_mode.h5")
lstm$run()
#lstm$save_model("models/final1.h5")

network.rmse <- lstm$rmse()
print(network.rmse)
pred <- lstm$predictions()
lstm$plot_predictions()


# ------------- Joining LSTM and ARIMA models
forecast.vector <- fc4 + pred

plot(testing.ts)
lines(forecast.vector, col=2)

rmse <- calc.rmse(forecast.vector)
print(rmse)



# ------------- Plot dependency of time
plot(abs(as.numeric(forecast.vector) - as.numeric(testing.ts)), type='l')





