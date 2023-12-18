import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from joblib import dump, load
import sklearn
from sklearn.linear_model import LogisticRegression
import base64



def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
            .stApp {{
                background-image: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
                background-size: cover;  /* Cover the entire screen */
                background-repeat: no-repeat;  /* Prevent the image from repeating */
                background-attachment: fixed;  /* Optional: Keep the background fixed during scrolling */

            }}
        </style>
      """,
      unsafe_allow_html=True,
      )
   


bg_img_url = 'Assets/bg.png'
sidebar_bg(bg_img_url)

#Create header
st.write("""# League of Legends Winner Predictor""")
st.write("""## How it works""")
st.write("Model your predicted winner by using the left side of the screen to apply value to the different metrics. This will give you a predicted winning team based on your selections. "
         "The current selections are default values.")

#image
image = Image.open('Assets/LOLland.jpg')
st.image(image)

#Create and name sidebar
# st.sidebar.header('Set your metrics')

# st.sidebar.write("""#### Choose your SG bias""")

def user_input_features():
    firstInhibitor = st.sidebar.slider('First Inhibitor', 0, 2, 0)  # Assuming binary value (0 or 1)
    t1_towerKills = st.sidebar.slider('T1 Tower Kills', 0, 11, 0)  # Assuming max 11 towers
    t1_inhibitorKills = st.sidebar.slider('T1 Inhibitor Kills', 0, 3, 0)  # Assuming max 3 inhibitors
    t1_baronKills = st.sidebar.slider('T1 Baron Kills', 0, 5, 0)  # Assuming an arbitrary max of 5 barons
    t1_dragonKills = st.sidebar.slider('T1 Dragon Kills', 0, 5, 0)  # Assuming an arbitrary max of 5 dragons
    t1_riftHeraldKills = st.sidebar.slider('T1 Rift Herald Kills', 0, 2, 0)  # Assuming max 2 heralds
    t2_towerKills = st.sidebar.slider('T2 Tower Kills', 0, 11, 0)  # Assuming max 11 towers
    t2_inhibitorKills = st.sidebar.slider('T2 Inhibitor Kills', 0, 3, 0)  # Assuming max 3 inhibitors
    t2_baronKills = st.sidebar.slider('T2 Baron Kills', 0, 5, 0)  # Assuming an arbitrary max of 5 barons
    t2_dragonKills = st.sidebar.slider('T2 Dragon Kills', 0, 5, 0)  # Assuming an arbitrary max of 5 dragons
    t2_riftHeraldKills = st.sidebar.slider('T2 Rift Herald Kills', 0, 2, 0)  # Assuming max 2 heralds

    user_data = {
        'firstInhibitor': firstInhibitor,
        't1_towerKills': t1_towerKills,
        't1_inhibitorKills': t1_inhibitorKills,
        't1_baronKills': t1_baronKills,
        't1_dragonKills': t1_dragonKills,
        't1_riftHeraldKills': t1_riftHeraldKills,
        't2_towerKills': t2_towerKills,
        't2_inhibitorKills': t2_inhibitorKills,
        't2_baronKills': t2_baronKills,
        't2_dragonKills': t2_dragonKills,
        't2_riftHeraldKills': t2_riftHeraldKills
    }

    features = pd.DataFrame(user_data, index=[0])
    return features


df_user = user_input_features()

team1 = 0

st.write("## Please input current game metrics in the left sidebar:")
st.write("Your current input shows below:")
st.write(df_user)
st.write("### Based on your input metrics")

logistic_model = load('model_train/trained_models/logistic_model.joblib')
prediction = logistic_model.predict(df_user)
team1 += 1 if prediction[0] == 1 else 0
color = 'blue' if prediction[0] == 1 else 'red'
st.write("#### Logistic regression model predicts team :{}[{}] is the winner".format(color, prediction[0]))


decision_tree_model = load('model_train/trained_models/decision_tree_model.joblib')
prediction = decision_tree_model.predict(df_user)
team1 += 1 if prediction[0] == 1 else 0
color = 'blue' if prediction[0] == 1 else 'red'
st.write("#### Decision tree model predicts team :{}[{}] is the winner".format(color, prediction[0]))

random_forest_model = load('model_train/trained_models/random_forest_model.joblib')
prediction = random_forest_model.predict(df_user)
team1 += 1 if prediction[0] == 1 else 0
color = 'blue' if prediction[0] == 1 else 'red'
st.write("#### Random forest model predicts team :{}[{}] is the winner".format(color, prediction[0]))


naive_bayes_model = load('model_train/trained_models/naive_bayes_model.joblib')
prediction = naive_bayes_model.predict(df_user)
team1 += 1 if prediction[0] == 1 else 0
color = 'blue' if prediction[0] == 1 else 'red'
st.write("#### Naive bayes model predicts team :{}[{}] is the winner".format(color, prediction[0]))

gradient_boosting_model = load('model_train/trained_models/gradient_boosting_model.joblib')
prediction = gradient_boosting_model.predict(df_user)
team1 += 1 if prediction[0] == 1 else 0
color = 'blue' if prediction[0] == 1 else 'red'
st.write("#### Gradient boosting model predicts team :{}[{}] is the winner".format(color, prediction[0]))
# final = prediction[0]

mlp_model = load('model_train/trained_models/mlp_model.joblib')
prediction = mlp_model.predict(df_user)
team1 += 1 if prediction[0] == 1 else 0
color = 'blue' if prediction[0] == 1 else 'red'
st.write("#### Multilayer Perceptron model predicts team :{}[{}] is the winner".format(color, prediction[0]))

linear_svc_model = load('model_train/trained_models/linear_svc_model.joblib')
prediction = linear_svc_model.predict(df_user)
team1 += 1 if prediction[0] == 1 else 0
color = 'blue' if prediction[0] == 1 else 'red'
st.write("#### Linear Support Vector Machine model predicts team :{}[{}] is the winner".format(color, prediction[0]))


ovr_model = load('model_train/trained_models/ovr_model.joblib')
prediction = ovr_model.predict(df_user)
team1 += 1 if prediction[0] == 1 else 0
color = 'blue' if prediction[0] == 1 else 'red'
st.write("#### One-vs-the-rest model predicts team :{}[{}] is the winner".format(color, prediction[0]))


knn_model = load('model_train/trained_models/knn_model.joblib')
prediction = knn_model.predict(df_user)
team1 += 1 if prediction[0] == 1 else 0
color = 'blue' if prediction[0] == 1 else 'red'
st.write("#### K-Nearest Neighbor model predicts team :{}[{}] is the winner".format(color, prediction[0]))

st.write("### Based on all predictions given by 9 models above, the final prediction is:")
finalcolor = 'blue' if team1 > 4 else 'red'
st.write("#### Team :{}[{}]".format(finalcolor, 1 if team1 > 4 else 2))
# st.write("#### Team :blue[{}] is the winner".format(final))
