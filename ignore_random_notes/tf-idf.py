from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# http://www.ultravioletanalytics.com/blog/tf-idf-basics-with-pandas-scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer
ngramrange = (1,2)
tvec = TfidfVectorizer(min_df=.0025, max_df=.1, stop_words='english', ngram_range=ngramrange)
tvec_weights = tvec.fit_transform(data_filtered['plain_text'])
weights = np.asarray(tvec_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': tvec.get_feature_names(), 'weight': weights})
weights_df.sort_values(by='weight', ascending=False).head(20)