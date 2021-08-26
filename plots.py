import matplotlib.pyplot as plt
data['date'] = pd.to_datetime(data['date_created'])
data['year'] = data.date.map(lambda x: x.year)
data['year'] = data.year.astype(int)

plt.hist(data['year'])
plt.show()





# create plot for number of tokens per document




# create plot for number of tokens in document

