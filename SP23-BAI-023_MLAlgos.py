import pandas as pd 
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt 
from sklearn.metrics import ConfusionMatrixDisplay

#DataSet: Defaults of Credit Card Clients
df =pd.read_csv("F:\\PAI\\ML Project\\UCI_Credit_Card (1).csv")
print(df)
print(df['default.payment.next.month'].value_counts())

x = df.iloc[:, 1:-1] 
y = df["default.payment.next.month"]  

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42 )

#preprocessing

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Smote=SMOTE(sampling_strategy='minority',random_state=42)
X_train_smote,y_train_smote=Smote.fit_resample(X_train_scaled,y_train)
print(y_train_smote.value_counts())


#Random Forest without preprocessing

clf = RandomForestClassifier(n_estimators=100, random_state=42)

model = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with default parameters:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Random Forest with preprocessing and optimisation

parameter_grid = {
    'n_estimators': [100, 150],      
    'max_depth': [10, 15],             
    'min_samples_split': [2, 5],               
    'max_features': ['sqrt', 'log2'],      
    'bootstrap': [True, False]             
}

Rf = RandomForestClassifier()

#Random Forest using random search
random_search = RandomizedSearchCV(Rf, param_distributions= parameter_grid, n_iter=10, cv=3, scoring='accuracy', verbose=2)
random_search.fit(X_train_smote, y_train_smote)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

y_pred_tuned = random_search.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred_tuned))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))

rf_report = classification_report(y_test, y_pred_tuned, output_dict=True)
accuracy_rf = rf_report['accuracy']
precision_rf = rf_report['macro avg']['precision']
recall_rf = rf_report['macro avg']['recall']
f1_rf = rf_report['macro avg']['f1-score']

conf_matrix = confusion_matrix(y_test, y_pred_tuned)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels= df ['default.payment.next.month'].unique())
disp.plot()
plt.show()

#Random Forest using grid search
grid_search = GridSearchCV(estimator= Rf, param_grid=parameter_grid, scoring='accuracy', cv = 3) 
grid_search.fit(X_train_scaled, y_train_smote) 

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

y_pred_tuned = grid_search.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred_tuned))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))

#XgBoost without preprocessing

xgb = XGBClassifier( random_state = 42)
model = xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with default parameters:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#XGBoost with preprocessing and optimisation

parameter_grid = {
    'learning_rate': [0.1, 0.5, 0.7],      
    'n_estimators': [100, 200, 300],          
    'max_depth': [4, 6, 8],                
    'subsample': [0.6, 0.8, 1],              
    'colsample_bytree': [0.6, 0.7, 0.8]                           
}

#XGBoost using random search
random_search = RandomizedSearchCV(xgb, param_distributions= parameter_grid, n_iter=20, cv=2, scoring='accuracy' )
random_search.fit(X_train_smote, y_train_smote)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

y_pred_tuned = random_search.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred_tuned))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))

xgb_report = classification_report(y_test, y_pred_tuned, output_dict=True)
accuracy_xgb = xgb_report['accuracy']
precision_xgb = xgb_report['macro avg']['precision']
recall_xgb = xgb_report['macro avg']['recall']
f1_xgb = xgb_report['macro avg']['f1-score']

conf_matrix = confusion_matrix(y_test, y_pred_tuned)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels= df ['default.payment.next.month'].unique())
disp.plot()
plt.show()

#XGBoost using grid search

grid_search = GridSearchCV(estimator= xgb, param_grid=parameter_grid, scoring='accuracy', cv = 3, verbose=2) 
grid_search.fit(X_train_smote, y_train_smote) 

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

y_pred_tuned = grid_search.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred_tuned))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))

titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
random_forest_metrics = [accuracy_rf, precision_rf, recall_rf, f1_rf]
xgboost_metrics = [accuracy_xgb, precision_xgb, recall_xgb, f1_xgb]

figure, axis = plt.subplots(1, 2, figsize=(8, 8))
figure.suptitle("Comparison of Evaluation Metrics", fontsize=16)

axis[0, 0].bar(titles, random_forest_metrics, color='blue', width=0.4)
axis[0, 0].set_title("Random Forest", fontsize=12)
axis[0, 0].set_xlabel('Evaluation Metrics', fontsize=10)
axis[0, 0].set_ylabel('Values', fontsize=10)

axis[0, 1].bar(titles, xgboost_metrics, color='purple', width=0.4)
axis[0, 1].set_title("XGBoost", fontsize=12)
axis[0, 1].set_xlabel('Evaluation Metrics', fontsize=10)
axis[0, 1].set_ylabel('Values', fontsize=10)

plt.tight_layout()
plt.show()
