import polars as pl


def split_by_days():
    return (
        pl.read_csv("data/preprocessed/events_data.csv", parse_dates=True)
        .with_column(pl.col('DATE').cast(pl.Date))
        .filter(pl.col("DATE") >= pl.lit("YYYY-mm-dd").str.strptime(pl.Date, fmt="%F"))
        .groupby("DATE").agg(pl.count())
        .sort("DATE")
    ).to_pandas()


def split_by_hours():
    return (
        pl.read_csv("data/preprocessed/events_data.csv", parse_dates=True)
        .with_column(pl.col("DATE").dt.truncate("1h"))
        .filter(pl.col("DATE") >= pl.lit("YYYY-mm-dd 00"))
        .groupby("DATE").agg(pl.count())
        .sort("DATE")
    ).to_pandas()
