
# load data
data = pd.read_pickle("/Users/danielsaggau/PycharmProjects/pythonProject/data/opinions_data.pkl")

#subset to data that also entails actual opinions; remove page count = 0 / NaN
data = data[~data['page_count'].isnull()] # remove empty pages

text = data['plain_text']
author = data['author']
url = data['download_url']
html = data['html']
id = data['id']

text = data['plain_text'] # subset
# cleaning text file
text = text.replace("plain_text", " ")
text = text.replace("  ", " ")

text = text.str.replace("\n"," ")
text = text.str.replace("FILED", "")
text = text.str.replace("NOT FOR PUBLICATION", "")
text = text.str.replace("FOR PUBLICATION", "")
text = text.str.replace("UNITED STATES COURT OF APPEALS", "")
text = text.str.replace("U.S. COURT OF APPEALS", "")
text = text.str.replace("U .S. COURT OF APPEALS", "")
text = text.str.replace("U .S. COURT OF APPEALS", "")
text = text.str.replace("UNITED STATES OF AMERICA", "")
text = text.str.replace("\x0c","")
text = text.str.replace("\uf8fc","")
text = text.str.replace("\uf8fd","")
text = text.str.replace("\uf8fe", " ")
text = text.str.replace("FOR THE NINTH CIRCUIT", "")
text = text.str.replace("Appeal from the United States District Court", "")
text = text.str.replace("[*]","")
text = text.str.replace("\*\*","")
text = text.str.replace("\n\n","")
text = text.str.replace("MOLLY C. DWYER, CLERK", " ")
text = text.str.replace("U .S. C O U R T OF APPE ALS", " ")
text = text.str.replace("  ", " ")
data['length'] = text.str.len()


text.to_csv('/Users/danielsaggau/PycharmProjects/pythonProject/data/output.txt', sep='\n', index=False)
