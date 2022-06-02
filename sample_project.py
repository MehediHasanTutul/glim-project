import streamlit as st
import pandas as pd
import numpy as np
from pickle import load
from sklearn.metrics import confusion_matrix
from PIL import Image



# import matplotlib.pyplot as plt
#This is a workarround
im = Image.open('icon.jpg')
st.set_page_config(
    page_title="GLIM PROJECT",
    page_icon=im,
    layout="wide",
)

st.markdown(
     """
     <style>
     #MainMenu {visibility: hidden;}
     footer {visibility: hidden;}
     [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
         width: 450px;
       }
       [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
           width: 500px;
           margin-left: -500px;
        }
        </style>
        """,
        unsafe_allow_html=True)

# st.sidebar.markdown("""
# [Example CSV input file](https://raw.githubusercontent.com/MehediHasanTutul/glim-project/Boss/sample_input.csv)
# """)

# uploaded = st.sidebar.file_uploader("please upload only .csv file as input", type={"csv"})
st.write("**An intelligent online platform to predict adverse outcomes from cancer-related malnutrition using the GLIM criteria**")
genre = st.sidebar.radio(
     "Choose input features:",
     ('All GLIM Features', 'Subset of Features', 'Subset of Features without Muscle Mass'))

if genre=='All GLIM Features':
    input_feature = 'inputs_glim_1_21'
elif genre=='Subset of Features':
    input_feature = 'inputs_glim_subset'
else:
    input_feature = 'inputs_glim_subset_no_muscle_mass'
    
output_button= st.sidebar.radio(
     "Choose output type:",
     ('Patient Status', 'Unplanned Admission'))
if output_button=='Patient Status':
    output_type = 'patient_status'
else:
    output_type = 'unplanned_admission'
    

inputs_list = ['inputs_glim_1_21', 'inputs_glim_subset', 'inputs_glim_subset_no_muscle_mass']
# modelnames_list = ['inputs_glim_1_21', 'inputs_glim_subset', 'inputs_glim_subset_no_muscle_mass']
output_list = ['patient_status', 'unplanned_admission']

modelname = f'model_{input_feature}_output_{output_type}'
path = f'https://raw.githubusercontent.com/MehediHasanTutul/glim-project/Boss/{modelname}'

model = load(open(f'{modelname}.pkl', 'rb'))

inputs = {
    'inputs_glim_1_21':['GLIM1', 'GLIM2', 'GLIM3', 'GLIM4', 'GLIM5', 'GLIM6', 'GLIM7', 'GLIM8', 'GLIM9', 'GLIM10', 'GLIM11', 'GLIM12', 'GLIM13', 'GLIM14', 'GLIM15', 'GLIM16', 'GLIM17', 'GLIM18', 'GLIM19', 'GLIM20', 'GLIM21'],
    'inputs_glim_subset':['GLIM1', 'GLIM3', 'GLIM5', 'GLIM10', 'GLIM13', 'GLIM14', 'GLIM19'],
    'inputs_glim_subset_no_muscle_mass':['GLIM1', 'GLIM2', 'GLIM3', 'GLIM4', 'GLIM7', 'GLIM8', 'GLIM10', 'GLIM11', 'GLIM16']
    }
#https://raw.githubusercontent.com/MehediHasanTutul/glim-project/Boss/
file_name = 'sample_input.csv'

st.sidebar.write('**Select Feature values:**')
c = st.sidebar.columns(5)   # 1,c2,c3,c4,c5
i=0
X = {}
for feat in inputs[input_feature]:
    X[feat]=[int(c[int(i%5)].radio(feat,(0,1)))]
    i = i+1
X= pd.DataFrame(X)


# st.sidebar.write('**Selected Features:**')
# st.sidebar.markdown(inputs[input_feature])

# @st.cache()
# def rescale_output(file_name):
#     df_main = pd.read_csv(file_name)
#     if 'patient_status' in df_main.columns:
#         df_main.loc[df_main['patient_status']==1,'patient_status']=0
#         df_main.loc[df_main['patient_status']==2,'patient_status']=1
       
#     return df_main


y=None

# if uploaded is None:
#     df_main = rescale_output(file_name)
#     st.write('**Awaiting CSV file to be uploaded. Currently showing output using sample input.**')
# else:
#     df_main = rescale_output(uploaded)
    
# df_main = rescale_output(file_name)
# if output_type in df_main.columns:
#     y = df_main[output_type].to_numpy()

# # 
# X = df_main[inputs[input_feature]]
y_pred_nb = model.predict(X.to_numpy())

y_pred = np.empty_like(y_pred_nb)
if output_type=='patient_status':
    y_pred[y_pred_nb==1]=2
    y_pred[y_pred_nb==0]=1
    
col1, col2 = st.columns(2)

df = pd.DataFrame({output_type:y_pred},index=[np.linspace(1,len(y_pred_nb),len(y_pred_nb))])

# cont = st.container()
st.write('**Selected Inputs:**')
st.write(X)
with col1:
    st.write('**Prediction:**')
    st.write(df)

if y is not None and len(y_pred_nb)>10:
    tn, fp, fn, tp = confusion_matrix(y, y_pred_nb).ravel()
    acc = round((tp+tn)/(tp+tn+fp+fn), 2)
    sen = round(tp/(tp+fn), 5) # recall
    spe = round(tn/(tn+fp), 3)
    pre = round(tp/(tp+fp), 3)
    
    df_tmp = pd.DataFrame({'accuracy':[acc], 'sensitivity':[sen], 'specificity': [spe], 'precision':[pre]})
    '**results in summary:**'
    df_tmp
    with col2:
        st.write('**confusion matrix:**')
        df = pd.DataFrame({'0':[tp,fn], '1':[fp,tn]},index=['0','1'])
        st.write(df)






if st.checkbox('show variable importance'):
    st.image(Image.open(f'{modelname}_var_imp.png'))

if st.checkbox('show tree structure'):
    
    st.image(Image.open(f'{modelname}_tree.png'))
