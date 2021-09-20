import pandas as pd
import re
import numpy
import statistics as st

#this file explore the basic statistic info about review text and the corresponding movie

#get the reviewId file
reviewId_list = pd.read_csv('./movie_merged_with_reviewId.csv')

#get the reviewId with the matched movies
review_list = reviewId_list['reviewId'].tolist()

#match the reviewId with the review text in the knowledge base file
review_text = pd.read_csv('./review_knowledge_base.csv')

#initialize a result dataframe
result_df = pd.DataFrame(columns=['index','review'])

#fill in the result pd by index
for rid in review_list:
    if str(rid) != "nan":
        tmp = review_text.loc[[rid]]

        result_df = result_df.append({'index':rid, 'review':tmp['review'].values}, ignore_index=True)

# #stire the result to file
result_df.to_csv('./matched_review_knowledge_id_text.csv')

################################################################################

result_df = pd.read_csv('./matched_review_knowledge_id_text.csv')

#prepare data cleaning
review_list = result_df["review"].tolist()
new_sen = []
replaced_review = []

#process data cleaning step: to lower, erase special symbol
for review in review_list:
    for word in review.split(" "):
        word = word.lower()
        match_pattern = re.findall(r'\b[a-z]{1,15}\b', word)
        new_sen.append(''.join(match_pattern))
    replaced_review.append(' '.join(new_sen))
    new_sen = []

#add back into dataframe
result_df['replaced_review_text'] = replaced_review

#count length of review
result_df["review_text_len"] = [len(x.split()) for x in result_df['replaced_review_text'].tolist()]

#drop unnecessary column
result_df = result_df.drop(columns=['Unnamed: 0','review'])
print(result_df.head())

# text_len = result_df['review_text_len'].tolist()
# print(st.stdev(text_len))
# print(st.mean(text_len))
# print(min(text_len))
# print(max(text_len))

#store the rsult to file
result_df.to_csv('./matched_review_knowledge_id_text.csv')