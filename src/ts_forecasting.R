library(reticulate)
library(ggplot2)
library(forecast)
library(tseries)
source_python('utils/split_data.py')

# ------------- Get T.S.
df <- split_by_days()
data.ts <- ts(df$count, start=YYYY, frequency=365.25)
clean.ts <- tsclean(data.ts)


# ------------- Train/Test split
# Train set: 2018--2020 years
train.ts <- ts(clean.ts[1:1096], start=YYYY, frequency=365.25)

# Test set: remaining 2021 years
testing.ts <- ts(clean.ts[1097:length(clean.ts)], start=YYYY+x, frequency=365.25)

# Train / test sets for multiseasonal time series
train.multi <- msts(train.ts, seasonal.periods=c(7, 365), start=YYYY)
test.multi <- msts(testing.ts, seasonal.periods=c(7, 365), start=YYYY+x)

# Train / test for holt winters
train.hw <- ts(train.ts, start=1, frequency=7)
test.hw <- ts(testing.ts, start=1, frequency=7)



calc.rmse <- function(predictions) {
    sqrt(mean((as.numeric(testing.ts) - as.numeric(predictions))^2))
}

# ------------- Seasonal naive forecast
model1 <- snaive(train.ts, h=length(testing.ts))
plot(testing.ts)
lines(model1$mean,col=2)

rmse1 <- calc.rmse(model1$mean)
## rmse1 210.0392

# ------------- Multiseasonal naive forecast
model1m <- snaive(train.multi, h=length(test.multi))
plot(test.multi)
lines(model1m$mean,col=2)

rmse1m <- calc.rmse(model1m$mean)
## rmse1 169.2033


# ------------- Random walk with drift forecast
model2 <- rwf(train.ts, drift=TRUE, h=length(testing.ts))
plot(testing.ts)
lines(model2$mean,col=2)

rmse2 <- calc.rmse(model2$mean)
## rmse2 234.9743


# ------------- Exponential smoothing (Holts) forecast
model3 <- holt(train.ts, seasonal="additive", h=length(testing.ts))
plot(testing.ts)
lines(model3$mean,col=2)

rmse3 <- calc.rmse(model3$mean)
## rmse3 149.5669


# ------------- Holt-Winters forecast with weekly seasonality
model3hw <- hw(train.hw, seasonal="additive", h=length(testing.ts))
plot(testing.ts)
lines(model3hw$mean, col=2)

rmse3hw <- calc.rmse(model3hw$mean)
## rmse3hw 229.2885


# ------------- ARMA model forecasting
model4 <- auto.arima(train.ts, trace=FALSE, seasonal=TRUE, stationary=TRUE, stepwise=FALSE, max.order=10, parallel=TRUE, num.cores=30)
summary(model4)
##  Best model: ARIMA(3,0,5)

fc4 <- forecast(model4, h=length(testing.ts))$mean
plot(testing.ts)
lines(fc4, col=2)

rmse4 <- calc.rmse(fc4)
## rmse4 142.0939



# ------------- Plot all forecasts
png(filename="resources/forecasting.png", units = "cm", width = 35.35, height = 28.35, pointsize = 18, res = 300)
plot(testing.ts, main="Title", cex.main=1.5, cex.lab=1.5, cex.axis=1.5, xlab="Date", ylab="Events", lwd = 1)
lines(model1m$mean, col=2, lwd = 1.5) 
lines(model2$mean, col=3, lwd = 1.5)
lines(model3$mean, col=4, lwd = 1.5)
lines(fc4, col=7, lwd = 1.5)
legend("topright", c("Multiseasonal Naive", "RW with drift", "Holt's exponential", "ARIMA(3, 0, 5)"), col=c(2, 3, 4, 7), lty=1)
dev.off()



# ------------- Try Holts-Winters method with hourly data
dfh <- split_by_hours()
datah.ts <- ts(dfh$count, start=1, frequency=24)
cleanh.ts <- tsclean(datah.ts)

# Train set: 2018--2020 years
trainh.ts <- ts(cleanh.ts[1:(1096*24)], start=1, frequency=24)

# Test set: remaining 2021 years
testingh.ts <- ts(cleanh.ts[(1097*24):length(cleanh.ts)], start=1097, frequency=24)

# Goes negative...
fithw <- hw(cleanh.ts, seasonal="additive", h=length(testingh.ts))
plot(testingh.ts)
lines(ts(fithw$mean, start=1097, frequency=24), col=2)

# Plot residuals from all ts
plot(fithw, ylab="Title",
     PI=FALSE, type="o", fcol="white", xlab="Year")
lines(fitted(fithw), col="red", lty=2)
lines(fithw$mean, type="o", col="red")

