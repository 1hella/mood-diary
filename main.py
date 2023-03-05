import pandas as pd
import streamlit as st
import glob
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px

filenames = glob.glob('diary/*.txt')

entries = []
for filename in filenames:
    with open(filename, 'r') as file:
        filename = filename.split('\\')[1]
        date = filename.split('.')[0]
        entries.append({"date": date, "entry": file.read()})

analyzer = SentimentIntensityAnalyzer()

analyses = []
for entry in entries:
    analysis = analyzer.polarity_scores((entry['entry']))
    analysis['date'] = entry['date']
    analyses.append(analysis)

df = pd.DataFrame(analyses)
print(df.dtypes)
pos = px.line(x=df['date'], y=df['pos'], labels={"x": "date", "y": "positivity"})
neg = px.line(x=df['date'], y=df['neg'], labels={"x": "date", "y": "negativity"})

st.title("Diary Analysis")

st.subheader('Positive')
st.plotly_chart(pos)

st.subheader('Negative')
st.plotly_chart(neg)