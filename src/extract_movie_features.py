#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:54:14 2017

@author: kolassc
"""

import pandas as pd

# In[]
movies_df = pd.read_pickle('/home/kolassc/Desktop/ucsd_course_materials/CSE291/project/code/movies_data.pkl')

# In[]
movies_df = movies_df.head(2900)
# In[]

# In[]
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(stop_words = 'english',max_features=500)
text_features = tfidf_vect.fit_transform(movies_df['storyline'].values)

# In[]
movies_df['year2'] = movies_df['year'].apply(lambda x: int(int(x)/5)*5 if x else 0)

text_features_array = text_features.toarray()
movie_text_dict = {}

# In[]
j=0
for i in movies_df['movieId']:
     movie_text_dict[i] = text_features_array[j]
     j+=1 
