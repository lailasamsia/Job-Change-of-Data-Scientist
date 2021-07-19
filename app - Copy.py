# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 00:14:04 2021

@author: user
"""

import pandas as pd
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.write("""
         # Predict Job Change of Data Scientist
         * Data Source :[Job Change of Data Scientist](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists)
         * Data default using aug_train.csv, please upload aug_test.csv
         """)

st.sidebar.header('User Input Parameters')
uploaded_file = st.sidebar.file_uploader('Upload your csv file', type=['csv'])
if uploaded_file is not None:
   input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        city_development_index = st.sidebar.slider('City Development Index',
                                              0.44, 0.94, 0.78)  
        training_hours = st.sidebar.slider('Training Hours', 1, 336, 47)
        company_size = st.sidebar.selectbox('Company Size', ('50-99','<10', '10-49', 
                                                             '100-500',
                                                             '500-999', '1000-4999',
                                                             '5000-9999', '10000+'))
        relevent_experience = st.sidebar.selectbox('Relevent Experience', 
                                                   ('No relevent experience','Has relevent experience'
                                                    ))
        enrolled_university = st.sidebar.selectbox('Enrolled University',
                                                   ('no_enrollment', 
                                                    'Full time course', 
                                                    'Part time course'))
        education_level = st.sidebar.selectbox('Education Level', ('Graduate',
                                                                   'Masters', 'High School',
                                                                   'Phd', 'Primary School'))
        last_new_job = st.sidebar.selectbox('Last New Job', ('>4','never', '1', '2',
                                                             '3', '4'))
        company_type = st.sidebar.selectbox(' Company Type', ('Pvt Ltd', 'Funded Startup',
                                                              'Public Sector', 'Early Stage Startup',
                                                              'NGO', 'Other'))
        experience = st.sidebar.selectbox('Experience', ('15','<1', '1', '2' ,'3', '4',
                                                         '5', '6', '7', '8', '9', '10',
                                                         '11', '12', '13', '14', '16', '17',
                                                         '18','19', '20', '>20' ))
        major_discipline = st.sidebar.selectbox('Major', ('STEM',' Humanities',' Other',
                                                         'Business Degree',
                                                         ' Arts',' No Major'))
        gender = st.sidebar.selectbox('Gender', ('Male', 'Female', 'Other'))
        

        data = {'city_development_index' : city_development_index,
            'training_hours' : training_hours,
            'company_size' : company_size,
            'relevent_experience' : relevent_experience,
            'enrolled_university' : enrolled_university,
            'education_level': education_level,
            'last_new_job' : last_new_job,
            'company_type' : company_type,
            'experience' : experience,
            'major_discipline' : major_discipline,
            'gender' : gender}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df= user_input_features()

df = pd.read_csv('aug_train.csv')
df =  pd.concat([input_df, df], axis=0)
target = df['target']

def preprocessing(df):
    df=df.drop(columns=['target', 'city', 'enrollee_id'])
   
    for col in df:
        if df[col].isnull().any(): 
            df[col] = df[col].fillna(df[col].mode()[0]) 
    df= df.replace({'relevent_experience': {'Has relevent experience': 1,
                                            'No relevent experience': 0}})
    nominal = ['gender','enrolled_university', 'education_level', 'major_discipline', 'experience',
           'company_size', 'company_type', 'last_new_job']
    for col in nominal:
        dummy = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df= pd.concat([df, dummy], axis=1)
        del df[col]
    return (df)


load_clf = pickle.load(open('classifier.pkl', 'rb'))


def preproc(df):
    df=df.drop(columns=['city', 'enrollee_id'])
   
    for col in df:
        if df[col].isnull().any(): 
            df[col] = df[col].fillna(df[col].mode()[0]) 
    df= df.replace({'relevent_experience': {'Has relevent experience': 1,
                                            'No relevent experience': 0}})
    nominal = ['gender','enrolled_university', 'education_level', 'major_discipline', 'experience',
           'company_size', 'company_type', 'last_new_job']
    for col in nominal:
        dummy = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df= pd.concat([df, dummy], axis=1)
        del df[col]
    return (df)

df = preprocessing(df)
df= df[:1]
prediction = load_clf.predict(df)
predict_proba =  load_clf.predict_proba(df)    


st.subheader('User Input Features')
if uploaded_file is not None:
    df = input_df
    pp_df = preproc(df)
    st.write(pp_df)
else:
    st.write('Awaiting for csv file to be uploaded')
    st.write(df)


st.subheader('Prediction')
target = np.array([0,1])
if uploaded_file is not None: 
        prediction = load_clf.predict(preproc(df))
        df['target'] = prediction
        pred_df= df.loc[:, ['enrollee_id', 'target']]
        st.write(pred_df)
else:
     st.write(prediction)

if uploaded_file is not None:
        df.drop(columns=['target'], inplace=True)
        predict_proba =  load_clf.predict_proba(preproc(df))
        predict_proba = pd.DataFrame(predict_proba)
        predict_proba['enrollee_id']= df['enrollee_id']
        predict_proba = predict_proba[['enrollee_id',0, 1]]
        st.write(predict_proba)
else:
    st.write(predict_proba)

st.subheader('Data Analysis Visualization')
import streamlit.components.v1 as components
def main():
	html_temp = """<div class='tableauPlaceholder' id='viz1626668724686' style='position: relative'><noscript><a href='#'><img alt='Job Change of Data Science ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Jo&#47;JobChange_16266730438960&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='JobChange_16266730438960&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Jo&#47;JobChange_16266730438960&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1626668724686');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='800px';vizElement.style.height='3427px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='800px';vizElement.style.height='3427px';} else { vizElement.style.width='100%';vizElement.style.height='2927px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
	components.html(html_temp)
if __name__ == "__main__":
	main()