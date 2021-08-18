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

