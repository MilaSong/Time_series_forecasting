library(reticulate)
library(ggplot2)
library(forecast)
library(tseries)
source_python('utils/split_data.py')

# ------------- Get T.S
df <- split_by_days()
data.ts <- ts(df$count, start=YYYY, frequency=365.25)


# ------------- Remove outliers
clean.ts <- tsclean(data.ts)
plot(data.ts, col=rgb(red = 1, green = 0, blue = 0, alpha = 0.5), alpha=0.25)
lines(clean.ts)


# ------------- Plot T.S
png(filename="resources/plot_ts.png", units = "cm", width = 25.35, height = 20.35, pointsize = 18, res = 300)
par(mfrow=c(3,1), mar = c(3,4,1,1.5))
plot(data.ts, xlab="", ylab="events", frame.plot=FALSE, cex.lab=1.5, cex.axis=1.5)
plot(data.ts - clean.ts, xlab="", ylab="data.ts - clean.ts", frame.plot=FALSE, cex.lab=1.5, cex.axis=1.5)
plot(clean.ts, xlab="Date", ylab="events", frame.plot=FALSE, cex.lab=1.5, cex.axis=1.5)
dev.off()


# ------------- Properties of T.S
# Simple properties
mean(clean.ts)
var(clean.ts)

# Autocorrelation function
## For weekly seasonality
png(filename="resources/acf_week.png", units = "cm", width = 25.35, height = 10.35, pointsize = 18, res = 300)
ggAcf(clean.ts, lag=37) + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 15))
dev.off()

## For yearly seasonality
png(filename="resources/acf_full.png", units = "cm", width = 25.35, height = 10.35, pointsize = 18, res = 300)
ggAcf(clean.ts) + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 15))
dev.off()


# Spectral density
spectrum(clean.ts)
spectrum(clean.ts, method="ar")
spectrum(clean.ts, log="no")
spectrum(clean.ts, method="ar", log="no")

# Ljung-Box Test
Box.test(clean.ts, lag = 1, type = "Ljung", fitdf=0)
Box.test(clean.ts, lag = 356, type = "Ljung", fitdf=0)

# Augmented Dickey-Fuller Test
adf.test(clean.ts)


# ------------- Components of T.S
# Simple decomposition of T.S
fits <- decompose(clean.ts, type="additive")
str(fits)
plot(fits)

fc <- forecast(fits$random+fits$trend, h=1095)
plot(fc)

# Simple decomposition of T.S
fitstl <- stl(clean.ts, s.window=7)
str(fitstl)
png(filename="resources/decomposition.png", units = "cm", width = 25.35, height = 20.35, pointsize = 18, res = 300)
plot(fitstl)
dev.off()

fc <- forecast(fitstl$time.series[,2] + fitstl$time.series[,3], h=1095)
png(filename="resources/decomp_forecast.png", units = "cm", width = 25.35, height = 12.35, pointsize = 18, res = 300)
par(mar=c(2,2,1,0))
plot(fc, main="", frame.plot=FALSE)
dev.off()


# ------------- Multiseasonal T.S. decomposition
x <- msts(clean.ts, seasonal.periods=c(7, 365), start=YYYY)
plot(x)

fitcomp <- mstl(x)
plot(fitcomp)


# ------------- Seasonality from periodogram
png(filename="resources/periodogram.png", units = "cm", width = 25.35, height = 20.35, pointsize = 18, res = 300)
decomp <- decompose(clean.ts, type="additive")
spectrum(decomp$seasonal+decomp$seasonal)
dev.off()
