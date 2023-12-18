# 6893BigData: RealTime Victory Predictor (RVP) for LoL 5v5 Matches

## Project Overview & Main Objective

**Objective**: The **RealTime Victory Predictor (RVP)** is an ML-based system designed to predict the outcomes (win/loss) of League of Legends (LoL) 5v5 matches in real time.

**Innovation**: Unlike traditional predictors focusing solely on player statistics, RVP emphasizes the impact of in-game objectives/events (e.g., first Baron, Dragon, tower) on match outcomes. This approach offers a fresh perspective on factors influencing the game results.

## Dataset

Data is sourced using the Riot API ([Riot Developer Portal](https://developer.riotgames.com/)), focusing on the North America region and ranked team 5v5 matches.

Data Collection Steps:
1. **Target Player Selection** (7k players): Challengers, Master, Grandmaster [**LEAGUE-V4**]
2. **Fetch Player's puuid** (7k players) [**SUMMONER-V4**]
3. **Recent Matches** (70k to 140k matches): 10 to 20 most recent matches per player [**MATCH-V5** by puuid]
4. **Match Information**: Remove duplicates and fetch detailed match info (50k to 110k matches) [**MATCH-V5** by matchid]

## Analytics

**Technologies Used**:
- Data preprocessing and analysis: **Python Pandas**
- Prediction models: **Scikit-learn**
- Ensemble of 9 models for final prediction
- Data visualization: **Seaborn**, **Plotly**, **Matplotlib**
- Interactive web page UI for live predictions: **Streamlit**

**Models**:
- Logistic Regression
- Decision Tree
- Random Forest
- Naive Bayes
- Gradient Boosting
- Multilayer Perceptron
- Linear Support Vector Machine
- One-vs-the-rest
- K-Nearest Neighbor

**System**:

- **Data Acquisition**: Retrieving match data from the Riot API, focusing on top-tier North American players.

- **Data Preprocessing**: Cleaning and structuring of data using Python Pandas for optimal model performance.

- **Model Training & Ensemble**: Training diverse machine learning models with Scikit-learn and combining them for improved accuracy.

- **Real-Time Prediction**: Capability to make instant predictions during live matches, adapting to ongoing game dynamics.

- **Interactive Web Interface**: Streamlit-powered platform for user interaction, allowing live input and instant prediction access.

**Visualization**:
- Histogram
- Pie Chart
- Correlation Heat Map

## Demo
![Project Website Screenshot](Assets/project.png)

## Languages

- **Python**
- **Streamlit**

## Contributors

- Zheyu Zhang
- Wen Song
