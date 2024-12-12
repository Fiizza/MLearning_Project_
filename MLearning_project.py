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
X_test = scaler.transform(X_test)

Pca=PCA(0.95)
X_Pca=Pca.fit_transform(X_Scaled)
print("Principal Components Shape: ", X_Pca.shape)

X_train,X_test,y_train,y_test=train_test_split(X_Pca,y,random_state=42)

Smote=SMOTE(sampling_strategy='auto',random_state=42)
X_train_sm,y_train_sm=Smote.fit_resample(X_train,y_train)
print(y_train_sm.value_counts())

Start_time=time.perf_counter()
light_gbm=lightgbm.LGBMClassifier(random_state=42)
light_gbm.fit(X_train_sm,y_train_sm)
pred=light_gbm.predict(X_test)
print("Classification Report: ")
clr_=classification_report(y_test,pred)
print(clr_)
param_grid={
    'learning_rate':[0.001,0.01,0.05,0.1],
    'objective':['binary'],
    'metric':['binary_error','logloss'],
    'num_leaves':[31,35,40],
    'force_col_wise':[True]
}
RandomizedSearch=RandomizedSearchCV(light_gbm,param_grid,cv=10,verbose=3,n_iter=10)
RandomizedSearch.fit(X_train_sm,y_train_sm)
print("Best Parameters for LightGBM: ", RandomizedSearch.best_params_)
best_model_=RandomizedSearch.best_estimator_
Best_Predictions_Rand=best_model_.predict(X_test)
clr=classification_report(y_test,Best_Predictions_Rand)
print("Classification Report: ")
print(clr)
GridSearch=GridSearchCV(light_gbm,param_grid,cv=10,verbose=2)
GridSearch.fit(X_train_sm,y_train_sm)
print("Best Parameters for LightGBM: ", GridSearch.best_params_)
best_model_=GridSearch.best_estimator_
Best_Predictions=best_model_.predict(X_test)
clr=classification_report(y_test,Best_Predictions)
print("Classification Report: ")
print(clr)
Accuracy=accuracy_score(y_test,Best_Predictions)
print("Accuracy: ",Accuracy)
Precision=precision_score(y_test,Best_Predictions,average='weighted')
print("Precision: ",Precision)
Recall=recall_score(y_test,Best_Predictions,average='weighted')
print("Recall: ",Recall)
F1_Score=f1_score(y_test,Best_Predictions,average='weighted')
print("F1-Score: ",F1_Score)
End_time=time.perf_counter()
Elapsed_time=End_time-Start_time
print("Execution Time: ", (Elapsed_time), 'secs')

Start_time_=time.perf_counter()
SVm=SVC(kernel='rbf',random_state=5,C=1.0)
SVm.fit(X_train_sm,y_train_sm)
y_pred=SVm.predict(X_test)
clrp=classification_report(y_test,y_pred)
print("Classification Report: ")
print(clrp)
param_grid={
    'kernel':['linear','rbf','poly','sigmoid'],
    'C':[0.01,0.1,1.0],
    'gamma':['scale','auto',0.01,0.1,1.0],
    'degree':[2,3,4,5]
}
RandomizedSearch_=RandomizedSearchCV(SVm,param_grid,cv=3,verbose=3,n_iter=2)
RandomizedSearch_.fit(X_train_sm,y_train_sm)
print("Best Parameters for SVM: ", RandomizedSearch_.best_params_)
best_model=RandomizedSearch_.best_estimator_
Best_predictions_=best_model.predict(X_test)
clr=classification_report(y_test,Best_predictions_)
print("Classification Report: ")
print(clr)
Accuracy_=accuracy_score(y_test,Best_Predictions)
print("Accuracy: ",Accuracy_)
Precision_=precision_score(y_test,Best_predictions_,average='weighted')
print("Precision: ",Precision_)
Recall_=recall_score(y_test,Best_predictions_,average='weighted')
print("Recall: ",Recall_)
F1_Score_=f1_score(y_test,Best_predictions_,average='weighted')
print("F1-Score: ",F1_Score_)
End_time_=time.perf_counter()
Elapsed_time_=End_time_-Start_time_
print("Exectution time for SVM: ",(Elapsed_time_),'secs'  )
GridSearch_=GridSearchCV(SVm,param_grid,cv=2,verbose=3,)
GridSearch_.fit(X_train_sm,y_train_sm)
print("Best Parameters for SVM: ", GridSearch_.best_params_)
best_model=GridSearch_.best_estimator_
Best_predictions_Gr=best_model.predict(X_test)
clr=classification_report(y_test,Best_predictions_Gr)
print("Classification Report: ")
print(clr)

Titles=['Accuracy','Precision','Recall','F1-Score']
lightgbm_metrics=[Accuracy,Precision,Recall,F1_Score]
Svm_metrics=[Accuracy_,Precision_,Recall_,F1_Score_] 
figure, axis=plt.subplots(2,2, figsize=(8,8))
figure.suptitle("Comparison of Evaluation metrics for Different Algorithms", fontsize=10)
axis[0,0].bar(Titles,lightgbm_metrics,color='Blue', width=0.3)
axis[0,0].set_title("LightGbm",fontsize=7)
axis[0,0].set_xlabel('Evaluation Metrics', fontsize=5)
axis[0,0].set_ylabel('Values',fontsize=5)
axis[0,1].bar(Titles,Svm_metrics,color='Purple', width=0.3)
axis[0,1].set_title("SVM",fontsize=7)
axis[0,1].set_xlabel('Evaluation Metrics', fontsize=5)
axis[0,1].set_ylabel('Values',fontsize=5)
plt.tight_layout()
plt.show()

confusionMatrix_LightGbm=confusion_matrix(y_test,Best_Predictions)
print("Confusion Matrix: ")
print(confusionMatrix_LightGbm)
Display_LightGbm_ConfMatrix=ConfusionMatrixDisplay(confusion_matrix=confusionMatrix_LightGbm, display_labels=df_defaults['default.payment.next.month'].unique())
Display_LightGbm_ConfMatrix.plot()
plt.show()
confusionMatrix_SVM=confusion_matrix(y_test,Best_predictions_)
print("Confusion Matrix: ")
print(confusionMatrix_SVM)
Display_Svm_ConfMatrix=ConfusionMatrixDisplay(confusion_matrix=confusionMatrix_SVM, display_labels=df_defaults['default.payment.next.month'].unique())
Display_Svm_ConfMatrix.plot()
plt.show()

#Random Forest

clf = RandomForestClassifier(n_estimators=100, random_state=42)

model = clf.fit(X_train_sm, y_train_sm)

y_pred = clf.predict(X_test)
accuracy = model.score(X_test, y_pred)
print("Accuracy with default parameters:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

parameter_grid = {
    'n_estimators': [100, 150],      
    'max_depth': [10, 15],             
    'min_samples_split': [2, 5],       
    'min_samples_leaf': [1, 2],         
    'max_features': ['sqrt', 'log2'],      
    'bootstrap': [True, False]             
}

Rf = RandomForestClassifier()

#using random search
random_search = RandomizedSearchCV(Rf, param_distributions= parameter_grid, n_iter=10, cv=2, scoring='accuracy', verbose=2)
random_search.fit(X_train_sm, y_train_sm)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

y_pred_tuned = random_search.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred_tuned))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))

#using grid search
grid_search = GridSearchCV(estimator= Rf, param_grid=parameter_grid, scoring='accuracy', cv = 3, verbose=2) 
grid_search.fit(X_train_sm, y_train_sm) 

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

y_pred_tuned = grid_search.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred_tuned))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))

conf_matrix = confusion_matrix(y_test, y_pred_tuned)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels= df ['default.payment.next.month'].unique())
disp.plot()
plt.show()

roc_auc = roc_auc_score(y_test, y_pred_tuned)
print(f"ROC AUC: {roc_auc}")


rf_report = classification_report(y_test, y_pred_tuned, output_dict=True)
accuracy_rf = rf_report['accuracy']
precision_rf = rf_report['macro avg']['precision']
recall_rf = rf_report['macro avg']['recall']
f1_rf = rf_report['macro avg']['f1-score']
f1_roc = roc_auc

#XgBoost

xgb = XGBClassifier( random_state = 42)
model = xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
accuracy = model.score(X_test, y_pred)
print("Accuracy with default parameters:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_tuned)
print(f"ROC AUC: {roc_auc}")

parameter_grid = {
    'learning_rate': [0.01, 0.05, 1],      
    'n_estimators': [100, 150, 200],          
    'max_depth': [5, 6, 7, 8],                
    'subsample': [0.6, 0.7, 0.8],              
    'colsample_bytree': [0.6, 0.7, 0.8]                           
}

#using random search
random_search = RandomizedSearchCV(xgb, param_distributions= parameter_grid, n_iter=20, cv=2, scoring='accuracy', verbose=2, n_jobs=-1 )
random_search.fit(X_train_sm, y_train_sm)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

y_pred_tuned = random_search.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred_tuned))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))

roc_auc = roc_auc_score(y_test, y_pred_tuned)
print(f"ROC AUC: {roc_auc}")

#using grid search

grid_search = GridSearchCV(estimator= xgb, param_grid=parameter_grid, scoring='accuracy', cv = 3, verbose=2) 
grid_search.fit(X_train_sm, y_train_sm) 

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

y_pred_tuned = grid_search.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred_tuned))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))

conf_matrix = confusion_matrix(y_test, y_pred_tuned)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels= df ['default.payment.next.month'].unique())
disp.plot()
plt.show()

roc_auc = roc_auc_score(y_test, y_pred_tuned)
print(f"ROC AUC: {roc_auc}")

xgb_report = classification_report(y_test, y_pred_tuned, output_dict=True)
accuracy_xgb = xgb_report['accuracy']
precision_xgb = xgb_report['macro avg']['precision']
recall_xgb = xgb_report['macro avg']['recall']
f1_xgb = xgb_report['macro avg']['f1-score']
f2_roc = roc_auc

titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
random_forest_metrics = [accuracy_rf, precision_rf, recall_rf, f1_rf, f1_roc ]
xgboost_metrics = [accuracy_xgb, precision_xgb, recall_xgb, f1_xgb, f2_roc]

figure, axis = plt.subplots(2, 2, figsize=(10, 8))
figure.suptitle("Comparison of Evaluation Metrics", fontsize=16)

axis[1, 0].bar(titles, random_forest_metrics, color='blue', width=0.4)
axis[1, 0].set_title("Random Forest", fontsize=12)
axis[1, 0].set_xlabel('Evaluation Metrics', fontsize=10)
axis[1, 0].set_ylabel('Values', fontsize=10)

axis[1, 1].bar(titles, xgboost_metrics, color='purple', width=0.4)
axis[1, 1].set_title("XGBoost", fontsize=12)
axis[1, 1].set_xlabel('Evaluation Metrics', fontsize=10)
axis[1, 1].set_ylabel('Values', fontsize=10)

plt.tight_layout()
plt.show()

