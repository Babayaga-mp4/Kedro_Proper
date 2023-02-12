import pickle
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from xgboost import XGBClassifier
from vizplotly import pie, world_map, hist


df = pd.read_csv('train.csv')
df.drop('churn', axis = 1, inplace = True)
columns = list(df.columns)
features = dict.fromkeys(df.columns)
scratch = pd.DataFrame([dict.fromkeys(pd.get_dummies(df).columns)])
# model = lgb.Booster(model_file = 'raw.json')
model = XGBClassifier()
model.load_model("xgb_pycharm.json")
mm = pickle.load(open('scaler.sav', 'rb'))
# print((list(model.feature_name())))


st.set_page_config(layout = "wide")
# col1, col2 = st.columns([1,1])


# col2.plotly_chart(pie(), use_container_width = False)
st.title('Probability Density: Features vs Churn')


option = st.selectbox( 'Input Features', tuple(df.columns))

# with st.sidebar:
if option:
        st.plotly_chart(hist(option) ,use_container_width = True)




@st.cache
def predict(features):
    features = pd.DataFrame([features])
    features = pd.get_dummies(features)
    cols_when_model_builds = model.get_booster().feature_names
    # cols_when_model_builds = list(model.feature_names_in_)
    cols = scratch.columns.difference(features.columns)
    new_features = pd.DataFrame(pd.concat([features, scratch[cols]], axis = 1)).fillna(0)
    new_features = new_features[cols_when_model_builds]
    new_features.to_csv('colab_sample_negativeChurn.csv')
    new_features = pd.DataFrame(mm.transform(new_features))
    new_features.columns = cols_when_model_builds
    # print('new_cols:', (list(new_features.columns)))

    return model.predict(new_features), model.predict_proba(new_features)

# Negative Churn Starter

st.title('The Model in ACTION!')
st.header('Enter the feature values')
features['state'] = st.selectbox("The customer's state", list(df['state'].unique()))
features['total_day_minutes'] = st.slider("Total time spoken during the day", 0.0, 400.0, 161.6)
features['number_customer_service_calls'] = st.slider("Total number of customer service calls", 0, 10, 1)
features['total_eve_minutes'] = st.slider("Total time spoken during the evening", 0.0, 359.0, 195.5)

original_title = '<p style="font-family:Courier; color:Blue; font-size: 20px;">'
st.header('More Inputs')
with st.expander("Advanced Features:"):
    features['account_length'] = st.slider("The customer's account duration", 0,245, 107)

    features['area_code'] = st.radio("The customer's area_code", list(df['area_code'].unique()), index = 0)
    features['international_plan'] = st.radio("Is the customer a subscriber to an International Plan?", ['yes', 'no'], index = 1)
    features['voice_mail_plan'] = st.radio("Is the customer a subscriber to a Voice Mail Plan?", ['yes', 'no'], index = 0)

    features['number_vmail_messages'] = st.slider("Number of voice mail messages", 0, 60, 26)
    features['total_day_calls'] = st.slider("Total number of calls during the day", 0, 201, 123)
    features['total_day_charge'] = st.slider("Total day charge", 0.0, 100.0, 27.47)
    features['total_eve_calls'] = st.slider("Total number of calls during the evening", 0, 170, 103)
    features['total_eve_charge'] = st.slider("Total evening charge", 0.0, 35.0, 16.62)
    features['total_night_minutes'] = st.slider("Total time spoken during the night", 0.0, 400.0, 254.4)
    features['total_night_calls'] = st.slider("Total number of calls during the night", 0, 175, 103)
    features['total_night_charge'] = st.slider("Total night charge", 0.0, 20.0, 11.45)
    features['total_intl_minutes'] = st.slider("Total international minutes", 0.0, 20.0, 13.7)
    features['total_intl_calls'] = st.slider("Total number of international calls", 0, 20, 3)
    features['total_intl_charge'] = st.slider("Total international charge", 0.0, 10.0, 3.7)

# hvar = """  <script>
#                     var_elements = window.parent.document.querySelectorAll('.streamlit-expanderHeader');
#                     elements[0].style.fontSize = 'x-large';
#                     elements[0].style.fontWeight = 'bold';
# <script>"""
# components.html(hvar, width = 0, height = 0)

# Positive Churn Starter
#
# st.title('The Model in ACTION!')
# st.header('Enter the feature values')
# features['state'] = st.selectbox("The customer's state", list(df['state'].unique()))
# features['total_day_minutes'] = st.slider("Total time spoken during the day", 0.0, 400.0, 129.1)
# features['number_customer_service_calls'] = st.slider("Total number of customer service calls", 0, 10, 4)
# features['total_eve_minutes'] = st.slider("Total time spoken during the evening", 0.0, 359.0, 228.5)
#
#
#
# with st.expander('Tweak Advanced Features:'):
#     features['account_length'] = st.slider("The customer's account duration", 0,245, 65)

    # features['area_code'] = st.radio("The customer's area_code", list(df['area_code'].unique()), index = 0)
    # features['international_plan'] = st.radio("Is the customer a subscriber to an International Plan?", ['yes', 'no'], index = 0)
    # features['voice_mail_plan'] = st.radio("Is the customer a subscriber to a Voice Mail Plan?", ['yes', 'no'], index = 1)

#     features['number_vmail_messages'] = st.slider("Number of voice mail messages", 0, 60, 0)
#     features['total_day_calls'] = st.slider("Total number of calls during the day", 0, 201, 137)
#     features['total_day_charge'] = st.slider("Total day charge", 0.0, 100.0, 21.95)
#     features['total_eve_calls'] = st.slider("Total number of calls during the evening" ,0 ,170 ,83)
#     features['total_eve_charge'] = st.slider("Total evening charge", 0.0, 35.0, 19.42)
#     features['total_night_minutes'] = st.slider("Total time spoken during the night", 0.0, 400.0, 208.8)
#     features['total_night_calls'] = st.slider("Total number of calls during the night", 0, 175, 111)
#     features['total_night_charge'] = st.slider("Total night charge", 0.0, 20.0, 9.40)
#
#     features['total_intl_calls'] = st.slider("Total number of international calls" ,0 ,20 ,6)
#     features['total_intl_minutes'] = st.slider("Total international minutes" ,0.0 ,20.0 ,12.7)
#     features['total_intl_charge'] = st.slider("Total international charge", 0.0, 10.0, 3.42)



if st.button('Predict'):
    response = list(predict(features))
    # print(response[0], (response))
    if response[0] == 0: st.success("This customer is NOT under the risk of churning."
                                    " Predicted with a confidence of {:.2f}%".format(float((response[1][0])[0])*100)
                                    )
    else: st.success("This customer is under the risk of churning."
                     " Predicted with a confidence of {:.2f}%".format(float((response[1][0])[1])*100)
                     )

st.title('Churn Rate across the States')
st.plotly_chart(world_map(), use_container_width = True)