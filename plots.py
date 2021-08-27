import matplotlib.pyplot as plt
import pandas as pd
data['date'] = pd.to_datetime(data['date_created'])
data['year'] = data.date.map(lambda x: x.year)
data['year'] = data.year.astype(int)

year = data['year']
plt.hist(data['year'], color ='xkcd:azure')
plt.title('Documents per year', fontdict=dict(size=14))
plt.gca().set(ylabel='Number of Documents', xlabel='Year')
plt.show()

# distibution word counts

plt.hist(data['length'], color = 'navy', bins = 1000)
plt.title('Distribution of Document Word Counts', fontdict=dict(size=14))
plt.gca().set(xlim=(0, 90000),ylim=(0,400), ylabel='Number of Documents', xlabel='Document Word Count')
plt.show()
#

well = data.groupby('year').length.mean()
well_2 = data.groupby('year')..sum()
plt.plot(well, color ='xkcd:azure')
plt.gca().set(xlim=(2010, 2021),ylim=(4000,17500), ylabel='Average Tokens', xlabel='Year')
fig = plt.figure()
ax = plt.axes()

#references: https://www.kaggle.com/gqfiddler/preliminary-analysis-and-topic-modeling?select=cleaning_functions.py

fig(figsize=(10,6))
plt.plot(yearly_counts.iloc[:46]) # index omits 2017(36) and 2018(1) which have not-yet-catalogued cases
plt.title('Number of cases per year with at least one attributed opinion (i.e., not decided per curiam)', fontsize=14)
plt.ylim((0,180))
plt.xlabel('Year')
plt.ylabel('Number of cases')
plt.show()



df = pd.DataFrame(data['per_curiam'])
plt.bar(data['per_curiam'])
df = df.apply(pd.value_counts)
