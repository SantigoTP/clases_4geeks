import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
import plotly.express as px
#Load Data
X_raw, y_raw = load_iris(return_X_y=True, as_frame=True)
df_raw = X_raw
df_raw["species"] = y_raw
#Preprocessing
df_baking = df_raw.copy()
df_baking.columns = df_baking.columns.str.lower().str.replace(" ","_").str.replace("(","").str.replace(")","")
df_baking["species"] = df_baking["species"].map({
    0:"setosa",
    1:"versicolor",
    2:"virginica"
})
df_baking["species"] = df_baking["species"].astype("category")
df=df_baking.copy()

#Web 
st.title("Iris dashboard")
st.write("Iris data-set table")

species = st.selectbox("Filter to:",['versicolor',"setosa","virginica"])

st.write(df[df["species"] == species])

my_plot = px.scatter_matrix(df, dimensions=["sepal_length_cm","sepal_width_cm", "petal_length_cm", "petal_width_cm"], color="species")
st.plotly_chart(my_plot)

st.markdown("Se observa la separación de clases de manera clara en las variables sepal_length y sepal_width, esto es los vectores $x_1, x_2$")