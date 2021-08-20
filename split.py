data = pd.read_csv('/Users/danielsaggau/PycharmProjects/pythonProject/okey.csv')

train, test = model_selection.train_test_split(
    data,
    test_size=0.2,
    random_state=42)
