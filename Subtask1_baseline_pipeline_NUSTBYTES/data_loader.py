import pandas as pd
df_eng_train = pd.read_csv("data/eng_clean.csv", encoding="utf-8-sig")
df_eng_train = df_eng_train[['text', 'polarization']]
df_eng_test= pd.read_csv("data/eng_test.csv", encoding="utf-8-sig")
df_eng_test = df_eng_test[['text', 'polarization']]

df_deu_train = pd.read_csv("data/deu_clean.csv", encoding="utf-8-sig")
df_deu_train = df_deu_train[['text', 'polarization']]
df_deu_test= pd.read_csv("data/deu_test.csv", encoding="utf-8-sig")
df_deu_test = df_deu_test[['text', 'polarization']]

df_spa_train = pd.read_csv("data/spa_clean.csv", encoding="utf-8-sig")
df_spa_train = df_spa_train[['text', 'polarization']]
df_spa_test= pd.read_csv("data/spa_test.csv", encoding="utf-8-sig")
df_spa_test = df_spa_test[['text', 'polarization']]
print(df_eng_train.head())
print(df_deu_train.head())
print(df_spa_train.head())