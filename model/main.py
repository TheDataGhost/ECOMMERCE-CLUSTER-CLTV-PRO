import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from imblearn.over_sampling import SMOTE

from sklearn.metrics import accuracy_score,classification_report

import pickle as pickle



def create_model(data):
    
    x = data.drop(['Segment'], axis=1)

    y = data['Segment']
 
    oversample = SMOTE(random_state=42)

    x_os, y_os = oversample.fit_resample(x, y)


    train_x,test_x,train_y,test_y = train_test_split(x_os,y_os,test_size = 0.30,random_state= 42)

    scaler = StandardScaler()

    train_x = scaler.fit_transform(train_x)

    test_x = scaler.transform(test_x)


    model = RandomForestClassifier(
        max_depth=5,
        min_samples_leaf=2,
        n_estimators=83,
        random_state=0)

    model.fit(train_x, train_y)


    ghost = model.predict(test_x)

    print('Accuracy_score : ',accuracy_score(test_y,ghost))

    print('Classification report : \n',classification_report(test_y,ghost))

    return model, scaler   





def get_clean_data():

  data = pd.read_csv("/Users/Dataghost/Machine Learning/The Ghostmode.csv")

  print(data.head())


  data.drop(['CustomerID','RFM_SCORE'],axis =1,inplace=True)
  
  data['Gender']=data['Gender'].map({'Male':1,'Female':0})

  for i in data.columns:

    if i != 'Segment' and data[i].dtype == 'object':

        label_encoder = LabelEncoder()

        data[i] = label_encoder.fit_transform(data[i])

  
  return data


def main():
 

  data = get_clean_data()

  print(data.info())


  model, scaler = create_model(data)

    # Dump the model to a pickle file
  with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Dump the scaler to a pickle file
  with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)




if __name__ == '__main__':
  main()