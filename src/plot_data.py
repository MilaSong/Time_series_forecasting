import plotly.express as px
import os
import sys
os.chdir(sys.path[0])
from utils.split_data import split_by_days

df = split_by_days()

fig = px.line(df, x='DATE', y="count")
fig.show()
