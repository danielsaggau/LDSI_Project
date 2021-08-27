
import random, os, glob, json
import pandas as pd
import time

import pandas.io.json as pd_json

random.seed(1)
fk = random.choices(os.listdir('/Users/danielsaggau/Downloads/ca9'), k =10000)
data = []
for fi in fk:
 with open(f'/Users/danielsaggau/Downloads/ca9/{fi}') as json_data:
     json_1 = json.load(json_data)
     data.append(pd.Series(json_1))

print(data)

data= pd.DataFrame(data)
print(data)

df.to.csv()