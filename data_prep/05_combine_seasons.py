import pandas as pd

df1 = pd.read_csv("gedi_valid_all_layers_from_h5_6_fall1.csv")
df2 = pd.read_csv("gedi_valid_all_layers_from_h5_6_spring_1.csv")
df3 = pd.read_csv("gedi_valid_all_layers_from_h5_6_summer_21000.csv")[:21000]

df1_len = df1.shape[0]
df2_len = df2.shape[0]
df3_len = df3.shape[0]
df2_copy = list(df2["id"].copy())
df3_copy = list(df3["id"].copy())
print(df1_len)
print(df2_len)
print(df3_len)
y = map(lambda x: x + df1_len, df2_copy)
df2["id"] = list(y)
y = map(lambda x: x + df1_len + df2_len, df3_copy)
df3["id"] = list(y)

df_new = pd.concat([df1,df2,df3], ignore_index=True)
print(len(df_new))
df_new.to_csv("gedi_valid_all_layers_from_h5_6_all.csv", index=False)