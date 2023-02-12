import pickle
import streamlit as st
import time
import streamlit.components.v1 as components
import pandas as pd
import os
from vizplotly import *


df = pd.read_csv('train.csv')
df2 = df.drop('churn', axis = 1)
columns = list(df2.columns)
features = dict.fromkeys(df2.columns)
scratch = pd.DataFrame([dict.fromkeys(pd.get_dummies(df2).columns)])

model = pickle.load(open('best_model.pkl', 'rb'))
mm = pickle.load(open('scaler.sav', 'rb'))

# df0 = pd.read_csv("test_0.csv")
# df1 = pd.read_csv("test_1.csv")
# df2 = pd.read_csv("test_2.csv")
# df3 = pd.read_csv("test_3.csv")
st.set_page_config(layout = "wide")









def preprocess(features):
    # features = pd.DataFrame([features])
    features = pd.get_dummies(features)
    # cols_when_model_builds = model.get_booster().feature_names
    cols_when_model_builds = list(model.feature_name_)
    print(cols_when_model_builds)
    cols = scratch.columns.difference(features.columns)
    new_features = pd.DataFrame(pd.concat([features, scratch[cols]], axis = 1)).fillna(0)
    new_features = new_features[cols_when_model_builds]
    new_features.to_csv('colab_sample_negativeChurn.csv')
    new_features = pd.DataFrame(mm.transform(new_features))
    new_features.columns = cols_when_model_builds
    return new_features


st.title('The Model in ACTION!')


# st.header("Choose a Dataframe")
# chosen_df = df0
#
# df_choice = st.selectbox("Select a dataframe", ["test_1", "test_2", "test_3", "test_4"])
#
# if df_choice == "test_1":
#     chosen_df = df0
# elif df_choice == "test_2":
#     chosen_df = df1
# elif df_choice == "test_3":
#     chosen_df = df2
# elif df_choice == "test_4":
#     chosen_df = df3

chosen_df = pd.DataFrame()

df_choice = st.file_uploader("Upload a file", type=["csv"])
if df_choice is not None: chosen_df = pd.read_csv(df_choice)

st.header("Selected Data")
st.dataframe(chosen_df, use_container_width = True)





if chosen_df.size:

    col1 ,col2 = st.columns([1 ,0.8], gap = 'large')

    col1.header("Predictions")
    processed_chosen_df = preprocess(chosen_df)
    Predictions = model.predict(processed_chosen_df)
    Confidence = pd.DataFrame(model.predict_proba(processed_chosen_df))
    chosen_df["Predictions"] = Predictions
    chosen_df["Confidence"] = Confidence[1]
    cols_to_move = ['Predictions' , 'Confidence']
    chosen_df = chosen_df[cols_to_move + [col for col in chosen_df.columns if col not in cols_to_move]]
    col1.header('')
    col1.dataframe(chosen_df)
    if col1.button('Save Predictions'):
        chosen_df.to_csv('Saved_Predictions_from_{}'.format(df_choice.name))
        col1.success('Saved as Saved_Predictions_from_{} at {}'.format(df_choice.name, os.getcwd()))

    col2.header("Distribution of the Classes")
    dist_option = col2.selectbox('Choose Distribution:', ['Cumulative', 'Segregated by Risk'])

    if dist_option == 'Cumulative':
        col2.plotly_chart(count_hist(chosen_df.rename(columns = {'Predictions': 'churn'} ,
                                              inplace = False)) ,use_container_width = True)
    else:
        temp_df = risk_segmentation(chosen_df)
        col2.plotly_chart(count_hist_risk(temp_df))


    col1.header("Filtering the Predictions")

    # with st.expander('Filters'):

    col1, col2 = st.columns([1,1], gap = "large")

    mini_value = col1.slider("Minimum Confidence Threshold" ,
                            float(chosen_df.Confidence.min()) ,float(chosen_df.Confidence.max()))

    maxi_value = col2.slider("Maximum Confidence Threshold" ,
                            min_value = float(chosen_df.Confidence.min()) ,max_value = float(chosen_df.Confidence.max()), value = 0.85)


    filtered_df = chosen_df[(chosen_df.Confidence >= mini_value) & (chosen_df.Confidence <= maxi_value)]

    st.dataframe(filtered_df)
    if st.button('Save Filtered Predictions'):
        chosen_df.to_csv('Filtered_Predictions_from_{}_min{}_max{}'.format(df_choice.name, mini_value, maxi_value))
        st.success('Saved as Filtered_Predictions_from_{}_min_{}_max_{} at {}'.format(df_choice.name, mini_value, maxi_value, os.getcwd()))



    st.title("Statistical Analysis")
    option = st.selectbox('Input Features' ,tuple(df.columns))

    col1 ,col2 = st.columns([1, 1] ,gap = 'large')

    # Add a check box to see Training data

    col1.header('Probability Density of Selected Data'
               )
    col2.header("Probability Density of Training Data")


    if option:
            col1.plotly_chart(hist(option, chosen_df.rename(columns={'Predictions': 'churn'},
                                                          inplace = False)), use_container_width = True)
            col2.plotly_chart(hist(option ,df) ,use_container_width = True)

    col1.header("Pair Plots of Selected Data")
    options = col1.multiselect("Input Features", tuple(df.columns))

    # Add a check box to see Training data

    col1.plotly_chart(advanced_stats(chosen_df.rename(columns={'Predictions': 'churn'},
                                                          inplace = False), options), use_container_width = True)
    col2.header("Pair Plots of Training Data")
    col2.header(" ")
    col2.header(" ")
    col2.plotly_chart(advanced_stats(df,options) ,use_container_width = True)
