import plotly.express as px
import pandas as pd
import json
df = pd.read_csv('train.csv')

def pie():
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


def world_map():
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
                               zoom=3.4, center = {"lat": 37, "lon": -95.7},
                               opacity=0.8,
                               labels={'churn':'churn_rate'}
                              )
    # fig.show()
    return fig




    # Distribution of Target vs the feature with most impact
def hist(option):
    fig = px.histogram(df, x=df[option], color="churn", marginal="violin", histnorm = 'probability density', barmode = 'group')
    # fig.show()
    return fig