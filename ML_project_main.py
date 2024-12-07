import pandas as pd

#DataSet: "Defaults of Credit Card Clients."
df_defaults=pd.read_csv("D:\\ML_Project\\Dataset\\UCI_Credit_Card.csv")
print(df_defaults)
print(df_defaults['default.payment.next.month'].value_counts())