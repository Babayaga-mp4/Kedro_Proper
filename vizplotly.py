import plotly.express as px
import pandas as pd
import json
import numpy as np
# df = pd.read_csv('train.csv')

def pie(df):
    # Pie Chart
    fig = px.pie(df, values=df.churn.value_counts(), names=['churn', 'no churn'], title = 'Distribution of the Target Variable')
    fig.update_traces(hoverinfo='label',
                      # textinfo='percent+label',
                      # textfont_size=20,
                      labels= ['Retention', 'Churn'],
                      # textposition='inside',
                      marker=dict(colors=['royalblue','Red'], line=dict(color='#000000', width=2)))
    # fig.show()
    return fig

    # Churn hotzones on US map


def world_map(df):
    with open('us-states.json') as response:
        counties = json.load(response)

    df2, df3 = df.copy(), df.copy()
    df2.churn.replace({"yes":1, "no":0}, inplace = True)
    df3.churn.replace({"yes":0, "no":1}, inplace = True)

    df2 = df2.groupby('state')['churn'].sum().to_frame().reset_index()
    df3 = df3.groupby('state')['churn'].sum().to_frame().reset_index()

    gb2 = dict(list( df2.groupby(['state'])))
    gb3 = dict(list( df3.groupby(['state'])))
    dummy = {}

    for state in gb2.keys():
        gb2[state]['churn'] = (gb2[state]['churn']/(gb2[state]['churn'] + gb3[state]['churn']))*100
        dummy[state] = float(str(gb2[state]['churn']).split()[1])


    dff = pd.DataFrame(dummy.items(), columns = ['state', 'churn'])
    fig = px.choropleth_mapbox(dff, geojson=counties, locations='state', color='churn',
                               color_continuous_scale="reds",
                               range_color=(5, 30),
                               mapbox_style="carto-positron",
                               zoom=3, center = {"lat": 39.8283, "lon": -98.5795},
                               opacity=0.75,
                               labels={'churn':'churn_rate'}

                              )
    # fig.update_layout(coloraxis_showscale = False)

    # fig.show()
    return fig




    # Distribution of Target vs the feature with most impact
def hist(option, df):
    fig = px.histogram(df, x=df[option], color="churn", marginal="violin", histnorm = 'probability density', barmode = 'group')
    # fig.show()
    return fig

def count_hist(df):
    positives = df[df['churn'] == 1].shape[0]
    negatives = df[df['churn'] == 0].shape[0]

    # Create a dataframe to hold the counts
    counts = pd.DataFrame({'Label': ['Churners' ,'Non-Churners'] ,
                           'Count': [positives ,negatives]})

    # Plot the histogram
    fig = px.bar(counts ,x = 'Label' ,y = 'Count' ,color = 'Label')
    return fig

def advanced_stats(df, options):
    df.churn.replace({1: "yes" ,0:"no"} ,inplace = True)
    fig = px.scatter_matrix(df ,
                            dimensions = options ,
                            color = "churn")
    return fig

def risk_segmentation(df):
    # df = df[df['Predictions'] == 1]
    conditions = [
        (df["Confidence"] <= 0.3) ,
        (df["Confidence"] > 0.3) & (df["Confidence"] <= 0.7) ,
        (df["Confidence"] > 0.6) & (df["Confidence"] <= 0.9) ,
        (df["Confidence"] > 0.9)
    ]

    # Define the values for the new column based on the conditions
    values = ["Low Risk" ,"Medium Risk" ,"High Risk" ,"Very High Risk"]

    # Create the new column in the DataFrame

    df["class"] = np.select(conditions ,values ,default = "Low Risk")

    return df

def count_hist_risk(df):
    class_counts = df["class"].value_counts()

    # Create variables for each class count
    class_1_count = class_counts["Low Risk"] if "Low Risk" in class_counts.index else 0
    class_2_count = class_counts["Medium Risk"] if "Medium Risk" in class_counts.index else 0
    class_3_count = class_counts["High Risk"] if "High Risk" in class_counts.index else 0
    class_4_count = class_counts["Very High Risk"] if "Very High Risk" in class_counts.index else 0

    # Create a dataframe to hold the counts
    counts = pd.DataFrame({'Label': ["Low Risk (<= 30%)" ,"Medium Risk (<= 70%)" ,
                                     "High Risk (<= 90%)" ,"Very High Risk (>90%)"] ,
                           'Count': [class_1_count, class_2_count, class_3_count, class_4_count]})

    # Plot the histogram
    colors = px.colors.sequential.Plasma

    fig = px.bar(counts ,x = "Label" ,y = "Count" ,color = "Label" ,title = "Number of Data Points in Each Class",
                 color_discrete_sequence=colors)
    # fig.update_traces(marker=dict(line=dict(color='red', width=1)))

    return fig