import streamlit as st
import pandas as pd
# import seaborn as sns
import plotly.express as px
import base64
from PIL import Image

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
   


bg_img_url = 'Assets/snow.jpg'
sidebar_bg(bg_img_url)






st.title("League of Legends Exploratory Data Analysis and Visualization")

st.markdown('''
A Web Page to visualize and analyze the League of Legends data from North America
* **Libraries Used:** Streamlit, Pandas, Plotly
* **Data Source:** Riot API
''')

# Reading csv data
data = pd.read_csv('model_train/match_data.csv')

# Create 3 tabs to the main panel of the web page.
tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Numeric Features", "Categorical Features"])

selected_attributes = [
    'winner',
    'firstInhibitor',
    't1_towerKills',
    't1_inhibitorKills',
    't1_baronKills',
    't1_dragonKills',
    't1_riftHeraldKills',
    't2_towerKills',
    't2_inhibitorKills',
    't2_baronKills',
    't2_dragonKills',
    't2_riftHeraldKills'
    ]

updated_data = data[selected_attributes]

with tab1:
    # Displaying Data and its Shape
    st.write("## Raw League of Legends Dataset", data)

    # extract meta-data from the uploaded dataset
    st.header("Meta-Data")
 
    row_count = data.shape[0]
 
    column_count = data.shape[1]
     
    # Use the duplicated() function to identify duplicate rows
    duplicates = data[data.duplicated()]
    duplicate_row_count =  duplicates.shape[0]
 
    missing_value_row_count = data[data.isna().any(axis=1)].shape[0]
 
    table_markdown = f"""
      | Description | Value | 
      |---|---|
      | Number of Rows | {row_count} |
      | Number of Columns | {column_count} |
      | Number of Duplicated Rows | {duplicate_row_count} |
      | Number of Rows with Missing Values | {missing_value_row_count} |
      """
 
    st.markdown(table_markdown)
    st.write(''' ''')

    st.write('''
    ## Feature Selection
    After considering the correlation matrix, we finally choose the following 11 features to explore 
    their relationship with the target variable 'the Winning Team'.
    * Target Variable: winner (the outcome variable indicating the winning team).
    * Feature Columns:
        1. firstInhibitor: Identifies which team destroyed the first inhibitor.
        2. t1_towerKills: The number of towers destroyed by team 1.
        3. t1_inhibitorKills: The number of inhibitors destroyed by team 1.
        4. t1_baronKills: The number of Barons killed by team 1.
        5. t1_dragonKills: The number of Dragons slain by team 1.
        6. t1_riftHeraldKills: The number of Rift Herald killed by team 1.
        7. t2_towerKills: The number of towers destroyed by team 2.
        8. t2_inhibitorKills: The number of inhibitors destroyed by team 2.
        9. t2_baronKills: The number of Barons killed by team 2. 
        10. t2_dragonKills: The number of Dragons killed by team 2.
        11. t2_riftHeraldKills The number of Rift Herald slain by team 2.
    ''')
    st.header("HeatMap")
    image = Image.open('Assets/heatmap.png')
    st.image(image)
    
    st.write("## Final Dataset", updated_data)

with tab2:
    # Processing the numerical features.
    num_features = ['gameDuration', 't1_towerKills', 't1_inhibitorKills', 't1_baronKills', 't1_dragonKills', 't1_riftHeraldKills', 't2_towerKills', 't2_inhibitorKills', 't2_baronKills', 't2_dragonKills', 't2_riftHeraldKills']
    
    selected_num_col = st.selectbox("Which numeric column do you want to explore?", num_features)

    st.header(f"{selected_num_col} - Statistics")
    col_info = {}
    col_info["Number of Unique Values"] = len(data[selected_num_col].unique())
    col_info["Number of Rows with Missing Values"] = data[selected_num_col].isnull().sum()
    col_info["Number of Rows with 0"] = data[selected_num_col].eq(0).sum()
    col_info["Number of Rows with Negative Values"] = data[selected_num_col].lt(0).sum()
    col_info["Average Value"] = data[selected_num_col].mean()
    col_info["Standard Deviation Value"] = data[selected_num_col].std()
    col_info["Minimum Value"] = data[selected_num_col].min()
    col_info["Maximum Value"] = data[selected_num_col].max()
    col_info["Median Value"] = data[selected_num_col].median()
    info_df = pd.DataFrame(list(col_info.items()), columns=['Description', 'Value'])
    # display dataframe as a markdown table
    st.dataframe(info_df)

    st.header("Histogram")
    fig = px.histogram(data, x=selected_num_col)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Process the categorical features.
    cat_features = ['winner', 'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald']
    # add select widget
    selected_cat_col = st.selectbox("Which text column do you want to explore?", cat_features)
    st.header(f"{selected_cat_col}")
    # add categorical column stats
    cat_col_info = {}
    cat_col_info["Number of Unique Values"] = len(data[selected_cat_col].unique())
    cat_col_info["Number of Rows with Missing Values"] = data[selected_cat_col].isnull().sum()
    cat_col_info["Number of Empty Rows"] = data[selected_cat_col].eq("").sum()
 
    cat_info_df = pd.DataFrame(list(cat_col_info.items()), columns=['Description', 'Value'])
    st.dataframe(cat_info_df)

    fig = px.pie(data, names=selected_cat_col)
    st.plotly_chart(fig, use_container_width=True)

# Custom CSS to enlarge tab text
custom_css = """
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.5rem;
    }
</style>
"""

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)