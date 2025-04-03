import os
import numpy as np
import pandas as pd
import uuid




df = pd.read_excel("/home/cillian/repos/JWOC_copy/uid.xlsx", engine="openpyxl", header=1)
df.head()
df.tail()

# Create unique identifier columns (not going to use uid1)
df['uid1'] = [uuid.uuid4() for _ in range(len(df.index))]
df['FN'] = df['FN'].str.lower().str.capitalize()
df['LN'] = df['LN'].str.lower().str.capitalize()
df['Unique Identifier'] = df['FN'].str[0:2] + "-" + df['LN'].str[0:2] + '-' + df['Sex']
#df['split'] = df['Name in Eng'].str.split(' ')
#df['uid2'] = df['split'].str[0].str[0:2] + "-" + df['split'].str[1].str[0:2] + '-' + df['Sex']
#df = df.drop('split', axis=1)
#df = df.drop('Unique Identifier', axis=1)
df

df.to_csv('/home/cillian/repos/JWOC_copy/cleanedUID.csv')
