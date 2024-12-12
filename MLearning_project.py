import pandas as pd
import matplotlib.pyplot as plt 
import time
import lightgbm 
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report,confusion_matrix,f1_score,precision_score,recall_score,accuracy_score,ConfusionMatrixDisplay
#DataSet: "Defaults of Credit Card Clients."
df_defaults=pd.read_csv("D:\\ML_Default_of_CreditCard_Clients\\Dataset_\\UCI_Credit_Card.csv")
print(df_defaults)
print(df_defaults['default.payment.next.month'].value_counts())
print(df_defaults.info())

X=df_defaults.loc[:,df_defaults.columns!='default.payment.next.month']
y=df_defaults['default.payment.next.month']

Scaler=StandardScaler()
X_Scaled=Scaler.fit_transform(X)

Pca=PCA(0.95)
X_Pca=Pca.fit_transform(X_Scaled)
print("Principal Components Shape: ", X_Pca.shape)

X_train,X_test,y_train,y_test=train_test_split(X_Pca,y,random_state=42)

Smote=SMOTE(sampling_strategy='auto',random_state=42)
X_train_sm,y_train_sm=Smote.fit_resample(X_train,y_train)
print(y_train_sm.value_counts())
