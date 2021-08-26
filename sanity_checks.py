desired_width=1020
pd.set_option('display.width', desired_width)
pd.set_option('display.max.columns',25)

#documents_by_id = {data['id']: d for d in data['plain_text']}
doc_lengths = [len(data['plain_text']) for id in data()]


data.info()
data.describe()

# sanity check: dates and case titles
checks = [83,1065,4508]
for c in checks:
    i = cases_df.index[c]
    print(
        '\n\n***SANITY CHECK {}***: \n',
        'CASE NAME:', cases_df.case_name[i], '\n',
        'CASE DATE:', cases_df.date_filed[i], '\n', '\n',
        'CASE TEXT:\n', cases_df.plain_text[i][:500])



# sanity check: length of each item


#plain_text = plain_text.str[700:]