# import main libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# import data set

df = pd.read_csv('../input/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv')


df.head(5)

# data information

df.info()

#we can see that some the features are object lets convert them into integer before that we will fill null values and make the data more worthable

# lets check does data contains null values 
df.isnull().sum()


# lets check how the values are deviated from mean and figure out how to fill bmi features nan values
df.describe()

# so we can fill nan values in bmi column with mean because data is in numeric and the values perfectly divated from mean 
df['bmi']=df['bmi'].fillna(df['bmi'].mean())


# lets convert strings into numeric format using label encoder

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

df['gender']=label.fit_transform(df['gender'])
df['ever_married']= label.fit_transform(df['ever_married'])
df['work_type']= label.fit_transform(df['work_type'])
df['Residence_type']= label.fit_transform(df['Residence_type'])
df['smoking_status']= label.fit_transform(df['smoking_status'])
df.head(5)

# droping id axis
df=df.drop('id', axis=1)

# visualise the data 

plt.figure(figsize =(7,5))
target = [len(df[df['stroke']==0]), len(df[df['stroke']==1])]
labels = ['no stroke', 'stroke']
colors = ['green','red']
explode =(0.05,0.1)

plt.pie(target,explode= explode,labels = labels, colors = colors,autopct='%4.2f%%', shadow = True,startangle=45)
plt.title('stroke percentage')

#we can clearly see that 95% is not effected by stroke and 5 percentage effected by stroke data set is not balanced

# lets visualize the dataset using bins 

plt.figure(figsize=(15,15))

for i, column in enumerate(df,1):
    plt.subplot(4,4,i)
    df[df['stroke']==0][column].hist(bins=35, color='blue',label='no stroke', alpha = 0.8)
    df[df['stroke']==1][column].hist(bins=35,color='red',label =' stroke', alpha =0.8)
    plt.legend()
    plt.xlabel(column)
    
    # lets see correlation with feaature using correlation function
    
    
cor = df.corr()
features = cor.index
plt.figure(figsize=(15,15))

heat = sns.heatmap(df[features].corr(),annot= True, cmap='RdYlGn')
plt.axis('equal')
plt.show()


# first we will train the data set

X= df.drop(['stroke'],axis=1)
y=df['stroke']
X



from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=25)



#smote  function

from imblearn.over_sampling import SMOTE
smote =SMOTE(random_state=2)
X_train,y_train=smote.fit_resample(X_train,y_train.ravel())

from xgboost import XGBClassifier

xg= XGBClassifier(objective='binary:logistic',max_depth=25,n_estimators=150,learning_rate =0.05,
                  eta=0.01,random_state=5,use_label_encoder=False,eval_metric='logloss')
xg= xg.fit(X,y)

predict =xg.predict(X_test)


xg.score(X_test,y_test)

# lets check model performance using f1 score 
from sklearn.metrics import f1_score,confusion_matrix, classification_report
f1= f1_score(y_test,predict)
f1


# lets check model performance using confusion matrix and classification report


cm = confusion_matrix(y_test,predict)
cm

print(classification_report(y_test,predict))



