import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pylab as plt

pd.set_option('display.max_columns', 25)


# ------ Define functions ------
def run_kmeans(n_clusters_f, init_f, df_f):
    k_means_model_f = KMeans(n_clusters=n_clusters_f, init=init_f).fit_predict(df_f)
    k_means_model_f = pd.DataFrame(k_means_model_f)
    k_means_model_f['123'] = df_f.index
    k_means_model_f = k_means_model_f.set_index('123', drop=True, append=False, inplace=False, verify_integrity=False)
    k_means_model_f.columns = ['predict_cluster_kmeans']

    df_f = pd.concat([df_f, k_means_model_f], axis=1, join='outer')

    # summarize cluster attributes
    k_means_model_f_summary = df_f.groupby('predict_cluster_kmeans').agg(attribute_summary_method_dict)
    return k_means_model_f, k_means_model_f_summary


# ------ Import data ------
df_sub = pd.read_csv('subscribers_clean.csv')
df_sub['age'].fillna(46)
df_sub['intended_use'].fillna('no_specific_intention')

# ----- variables to dummies -----
df_sub['male_TF'] = df_sub['male_TF'].astype(int)
# print(df_sub.head())
df_sub.columns
df_sub_dummies = pd.get_dummies(data=df_sub, columns=['preferred_genre', 'intended_use'])
# print(df_sub_dummies.columns)
# print(df_sub_dummies.shape)
# df_sub_dummies.head()

# ------ RUN CLUSTERING -----
# --- set parameters
n_clusters = 3
init_point_selection_method = 'k-means++'

# --- select data
##### specify list of attributes on which to base clusters
cols_for_clustering = ['age', 'male_TF',
                        'preferred_genre_comedy', 'preferred_genre_drama',
                        'preferred_genre_international', 'preferred_genre_no_preference',
                        'preferred_genre_other', 'preferred_genre_regional',
                        'intended_use_access to exclusive content', 'intended_use_education',
                        'intended_use_expand international access',
                        'intended_use_expand regional access', 'intended_use_other',
                        'intended_use_replace OTT', 'intended_use_supplement OTT']
df_cluster = df_sub_dummies.loc[:, cols_for_clustering]

print(df_cluster.isnull().any())
# --- split to test and train
# df_cluster_train, df_cluster_test, _, _, = train_test_split(df_cluster, [1]*df_cluster.shape[0], test_size=0.33)

# --- fit model
attribute_summary_method_dict = {'age': np.mean,'male_TF':sum,'preferred_genre_comedy':sum,'preferred_genre_drama':sum,
                                 'preferred_genre_international':sum,'preferred_genre_no_preference':sum,
                                 'preferred_genre_other':sum, 'preferred_genre_regional':sum,
                                 'intended_use_access to exclusive content':sum,
                                 'intended_use_education':sum,'intended_use_expand international access':sum,
                                 'intended_use_expand regional access':sum,'intended_use_other':sum,
                                 'intended_use_replace OTT':sum,'intended_use_supplement OTT':sum}
col_output_order = ['age','preferred_genre_comedy', 'preferred_genre_drama', 'preferred_genre_international',
                    'preferred_genre_no_preference', 'preferred_genre_other', 'preferred_genre_regional',
                    'intended_use_access to exclusive content', 'intended_use_education',
                    'intended_use_expand international access', 'intended_use_expand regional access',
                    'intended_use_other', 'intended_use_replace OTT', 'intended_use_supplement OTT']

# training data
# train_model, train_model_summary = run_kmeans(n_clusters, init_point_selection_method, df_cluster_train.reindex())
# testing data
# test_model, test_model_summary = run_kmeans(n_clusters, init_point_selection_method, df_cluster_test.reindex())

# all data
model, model_summary = run_kmeans(n_clusters, init_point_selection_method, df_cluster)

# --- run for various number of clusters
##### add the code to run the clustering algorithm for various numbers of clusters
ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(df_cluster)
    inertias.append(model.inertia_)

# --- draw elbow plot
##### create an elbow plot for your numbers of clusters in previous step
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

model = KMeans(n_clusters=3, n_init=10)
model.fit(df_cluster)
output = pd.DataFrame(model.cluster_centers_)
output.columns = df_cluster.columns
output.to_excel('Kmeans_output.xls')
