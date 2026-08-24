import numpy as np 
import pandas as pd 

from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import RandomOverSampler 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score 
from rich import print

import tensorflow as tf 


# data featuring
def feature_dataset(df,drop_num=True):
    if drop_num:
        df.drop(columns=['customerID','PaymentMethod'],inplace=True)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['gender'] = (df['gender']=='Male').astype(int) 
    df['Partner'] = (df['Partner']=='Yes').astype(int) 
    df['Dependents'] = (df['Dependents']=='Yes').astype(int) 
    df['PhoneService'] = (df['PhoneService']=='Yes').astype(int) 
    df['MultipleLines'] = df['MultipleLines'].apply(lambda x:0 if x=='Yes' else(1 if x=='No' else 2))
    df['InternetService'] = df['InternetService'].apply(lambda x:0 if x=='DSL' else(1 if x=='No' else 2))
    df['OnlineSecurity'] = df['OnlineSecurity'].apply(lambda x:0 if x=='Yes' else(1 if x=='No' else 2))
    df['OnlineBackup'] = df['OnlineBackup'].apply(lambda x:0 if x=='Yes' else(1 if x=='No' else 2))
    df['DeviceProtection'] = df['DeviceProtection'].apply(lambda x:0 if x=='Yes' else(1 if x=='No' else 2))
    df['TechSupport'] = df['TechSupport'].apply(lambda x:0 if x=='Yes' else(1 if x=='No' else 2))
    df['StreamingTV'] = df['StreamingTV'].apply(lambda x:0 if x=='Yes' else(1 if x=='No' else 2))
    df['StreamingMovies'] = df['StreamingMovies'].apply(lambda x:0 if x=='Yes' else(1 if x=='No' else 2))
    df['Contract'] = df['Contract'].apply(lambda x:0 if x=='One year' else(1 if x=='Two year' else 2))
    df['PaperlessBilling'] = (df['PaperlessBilling']=='Yes').astype(int) 
    df['Churn'] = (df['Churn']=='Yes').astype(int)
    return df 

# splitting dataset into train,validation and testing
def split_dataset(df):
    shuffled = df.sample(frac=1,random_state=42) 
    n = len(shuffled)
    train = shuffled[:int(0.6*n)]
    valid = shuffled[int(0.6*n):int(0.8*n)]
    test = shuffled[int(0.8*n):] 
    return train,valid,test 

# dataset scaling,imputing and fitting
def scale_dataset(dataframe,oversample=False,scaling=True):
    x = dataframe[dataframe.columns[:-1]].values 
    y = dataframe[dataframe.columns[-1]].values 

    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(x)

    if scaling:
        scaler = StandardScaler() 
        x = scaler.fit_transform(x) 
    if oversample:
        ros = RandomOverSampler(random_state = 42) 
        x,y = ros.fit_resample(x,y)  
    data = np.hstack((x,np.reshape(y,(-1,1)))) 
    return data,x,y 


# supervised learning model training session except neural net
def model_predict(model,train_x,train_y,test_x):
    model = model.fit(train_x,train_y) 
    y_pred = model.predict(test_x) 
    y_prob = model.predict_proba(test_x)[:, 1]
    return y_pred,y_prob

# neural net model training session
def neuralnet_model_predict(train_x,train_y,valid_x,valid_y,test_x,input_col):
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_col,)),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])
    nn_model.compile(
        optimizer = tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
    )
    nn_model.fit(
        train_x,
        train_y,
        validation_data = (valid_x,valid_y),
        epochs = 100,
        batch_size = 32,
        verbose = 0 
    )
    y_prob = nn_model.predict(test_x).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred , y_prob

def get_report(test_y,pred_y):
    str_report = classification_report(test_y,pred_y)
    li_report = str_report.split('\n')
    report = [] 
    for r,c in enumerate(li_report):
        if len(c)>0 and r>0:
            report.append(c.split()[1:])
    report[2].insert(0,'')
    report[2].insert(0,'')
    for r,c in enumerate(report[3:]):
        report[r+3] = c[1:]
    report = np.array(report) 
    report_df = pd.DataFrame(report,index=['0','1','accuracy','macro avg','weighted avg'],columns=['precision','recall','f1_score','support'])
    return report_df

def get_model_score(test_y,pred_y,prob_y):
    report = get_report(test_y,pred_y)
    accuracy = float(report['f1_score']['accuracy'])
    precision_0 = float(report['precision']['0'])
    precision_1 = float(report['precision']['1'])
    recall_0 = float(report['recall']['0'])
    recall_1 = float(report['recall']['1'])
    f1_score_0 = float(report['f1_score']['0'])
    f1_score_1 = float(report['f1_score']['1'])
    roc_auc = roc_auc_score(test_y, prob_y)  

    total_score = accuracy + precision_0 + precision_1 + recall_0 + recall_1 + f1_score_0 + f1_score_1 + roc_auc 
    # just for showing the metrics 
    print(f"accuracy : {accuracy}")
    print(f"precision_0 : {precision_0}")
    print(f"precision_1 : {precision_1}")
    print(f"recall_0 : {recall_0}")
    print(f"recall_1 : {recall_1}")
    print(f"f1_0 : {f1_score_0}")
    print(f"f1_1 : {f1_score_1}")
    print(f"f1_1 : {f1_score_1}")
    print(f"roc-auc : {roc_auc}")
    return total_score 

def get_supervised_model(df):
    train,valid,test = split_dataset(df) 

    train,train_x,train_y = scale_dataset(train,oversample=True)
    valid,valid_x,valid_y = scale_dataset(valid)
    test,test_x,test_y = scale_dataset(test) 

    knn_model = KNeighborsClassifier(n_neighbors=3) 
    nb_model = GaussianNB() 
    lg_model = LogisticRegression() 
    svm_model = SVC(probability=True) 
    s_model = None 
    max_score = 0.0
    models = [knn_model,nb_model,lg_model,svm_model]
    for model in models:
        print(f"----------------{model.__class__.__name__}------------------")
        y_pred,y_prob = model_predict(model,train_x,train_y,test_x) 
        total_model_score = get_model_score(test_y,y_pred,y_prob) 
        if(total_model_score > max_score):
            max_score = total_model_score 
            s_model = model 
    input_col = 18
    nn_ypred,nn_yprob = neuralnet_model_predict(train_x,train_y,valid_x,valid_y,test_x,input_col)
    nn_model_score = get_model_score(test_y,nn_ypred,nn_yprob) 
    if nn_model_score > max_score:
        max_score = nn_model_score
        s_model = "nn_model"
    return s_model,max_score


def get_pcaX(x):
    pca = PCA(n_components=5)
    transformed_x = pca.fit_transform(x)
    return transformed_x

def get_logistic_model_score(train_x,train_y,test_x,test_y):
    lg_model = LogisticRegression() 
    y_pred,y_prob = model_predict(lg_model,train_x,train_y,test_x) 
    print(f"=========={lg_model.__class__.__name__}=========")
    model_score = get_model_score(test_y,y_pred,y_prob) 
    return lg_model,model_score 

def get_unsupervised_model(df):
    train,valid,test = split_dataset(df) 
    train,train_x,train_y = scale_dataset(train,oversample=True,scaling=False)
    valid,valid_x,valid_y = scale_dataset(valid,scaling=False)
    test,test_x,test_y = scale_dataset(test,scaling=False)

    transformed_train_x = get_pcaX(train_x)
    transformed_test_x = get_pcaX(test_x)
    return get_logistic_model_score(transformed_train_x,train_y,transformed_test_x,test_y)
    

def get_model(df):
    supervised_model,total_model_score = get_supervised_model(df) 

    pca_logistic,model_score = get_unsupervised_model(df)
    model = None 
    model_type = None 
    if model_score > total_model_score:
        model = pca_logistic
        model_type = 'unsupervised' 
    else:
        model = supervised_model
        model_type = 'supervised'
    return model,model_type

def take_input(df):
    print("Look at this carefully:---")
    print("gender(Male/Female),SeniorCitizen(0/1),Partner(Yes/No),Dependents(Yes/No),tenure(a int numerical value),PhoneService(Yes/No),MultipleLines(Yes/No/No phone service),InternetService(DSL/No/Fiber optic),OnlineSecurity(Yes/No/No internet service),OnlineBackup(Yes/No/No internet service),DeviceProtection(Yes/No/No internet service),TechSupport(Yes/No/No internet service),StreamingTV(Yes/No/No internet service),StreamingMovies(Yes/No/No internet service),Contract(One year/Two year/Month-to-month),PaperlessBilling(Yes/No),MonthlyCharges(a float numerical value),TotalCharges(a float numerical value)") 
    print("Example : Male,0,Yes,Yes,20,Yes,No,Fiber optic,Yes,Yes,No internet service,No,No,No,Month-to-month,No,50.5,1350.75")
    x = input("Enter as the example : ") 
    x = x.split(',') 
    x.append("No")
    
    input_df = pd.DataFrame([x],columns=df.columns) 
    input_df = feature_dataset(input_df,drop_num=False)
    _,input_x,_ = scale_dataset(input_df)
    return input_x 

def output(model,data_x,model_type):
    y_pred = None 
    if model_type == 'supervised':
        if isinstance(model,str):
            y_prob = model.predict(data_x).flatten()
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            y_pred = model.predict(data_x) 
    elif model_type == 'unsupervised':
        transformed_data_x = get_pcaX(data_x)
        y_pred = model.predict(transformed_data_x)
    return y_pred


if __name__ == '__main__':
    df = pd.read_csv('data/telco.csv')
    df = feature_dataset(df) 
    model,model_type = get_model(df) 

    while True:
        print("------->> Type 1 for seeing model prediction or Type 0 for exit <<--------") 
        menu = int(input("Enter : ")) 
        if menu == 0:
            print("[turquoise1]Thank you! See you again![/]")
            break 
        elif menu == 1: 
            input_x = take_input(df)
            data_y = output(model,input_x,model_type)[0]
            out_str = None 
            if data_y == 0 :
                out_str = "[bold red]No[/bold red]" 
            else:
                out_str = "[green]Yes[/green]"
            print(f"[bright_blue]The model prediction is Churn[/] : {out_str}")
        else:
            print("You may type wrong input! Please type the correct menu number again : ") 
        
        
    