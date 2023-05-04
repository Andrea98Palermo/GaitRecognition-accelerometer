from collections import defaultdict
import math
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import euclidean_distances
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LogisticRegression
import math


seed = 28
np.random.seed(seed)
np.set_printoptions(threshold=sys.maxsize)

#implementazione PFA 
class PFA(object):

    def __init__(self, n_features, q=None):
        self.q = q
        self.n_features = n_features

    def fit(self, X):
        if not self.q:
            self.q = X.shape[1]

        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA(n_components=self.q, random_state=42).fit(X)                      #PCA
        A_q = pca.components_.T                                                     #componenti principali = colonne di A_q

        kmeans = KMeans(n_clusters=self.n_features, random_state=42).fit(A_q)       #raggruppamento righe di A_q con KMeans
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_                                   #valore medio di ogni cluster

        dists = defaultdict(list)
        for i, c in enumerate(clusters):                                            #scelta vettore più vicino a media per ogni cluster
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]
        

#lettura carattersitche e identità da csv
df = pd.read_csv("./features_zju.csv")
y = pd.read_csv("./targetVector.csv", header=None)
df.drop('id', axis=1, inplace=True)
y = pd.Series([x[0] for x in y.to_numpy()])

#selezione caratterstiche
df = VarianceThreshold().fit_transform(df, y)               #eliminazione feature costanti
df = SelectPercentile(percentile=45).fit_transform(df, y)   #eliminazione 55% con anova F-value
pfa = PFA(n_features=400)                                   #principal feature analysis
pfa.fit(df)
df = pfa.features_

#divisione sessioni
session_1 = df[0 : math.ceil(df.shape[0] / 2)]              
session_2 = df[math.ceil(df.shape[0] / 2) : ]
y1 = y[0 : math.ceil(df.shape[0] / 2)]
y2 = y[math.ceil(df.shape[0] / 2) : ]
y2.reset_index(inplace=True, drop=True)

sm = SMOTE(random_state=seed)
session_1, y1 = sm.fit_resample(session_1, y1)

clf = Pipeline([
    ('scaling', StandardScaler()),
    #('classification', AdaBoostClassifier(random_state=28, n_estimators=500))
    ('classification', LogisticRegression(random_state=28, max_iter=20000, solver="saga", multi_class="ovr"))
])
clf.fit(session_1, y1)                                                      #allenamento modello
clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')         #creazione modello con probabilità calibrate
clf_isotonic.fit(session_1, y1) 

y_scores = clf_isotonic.predict_proba(session_2)
ranks = {}
rates = []

#calcolo ranks
for i in range(len(session_2)):

    label = y2[i]
    scores = y_scores[i]
    index = np.where(clf_isotonic.classes_ == label)
    score = scores[index]

    ranks[label] = 1
    for x in scores:
        if(x > score):
            ranks[label] += 1


print("RANKS: ", len(ranks), "\n", ranks)

distinct_ranks = set(ranks.values())

print("DISTINCT RANKS: ", len(distinct_ranks), "\n", distinct_ranks)

#calcolo CMC
for rnk in distinct_ranks:
    prob = sum(1 for v in ranks.values() if v <= rnk) / len(ranks.values()) 
    rates.append((rnk, prob))


print("RATES: ", len(rates), "\n", rates)

x_val = [x[0] for x in rates]
y_val = [x[1] for x in rates]


plt.plot(x_val, y_val, 'ro')
plt.show()




#SECONDO ROUND CON SESSIONI INVERTITE

sm = SMOTE(random_state=seed)
session_2, y2 = sm.fit_resample(session_2, y2)

clf = Pipeline([
    ('scaling', StandardScaler()),
    ('classification', LogisticRegression(random_state=28, max_iter=20000, solver="saga", multi_class="ovr"))
])
clf.fit(session_2, y2)                                                      #allenamento modello
clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')         #creazione modello con probabilità calibrate
clf_isotonic.fit(session_2, y2) 

y_scores = clf_isotonic.predict_proba(session_1)
ranks = {}
rates = []

#calcolo ranks
for i in range(len(session_1)):

    label = y1[i]
    scores = y_scores[i]
    index = np.where(clf_isotonic.classes_ == label)
    score = scores[index]

    ranks[label] = 1
    for x in scores:
        if(x > score):
            ranks[label] += 1


print("RANKS: ", len(ranks), "\n", ranks)

distinct_ranks = set(ranks.values())

print("DISTINCT RANKS: ", len(distinct_ranks), "\n", distinct_ranks)

#calcolo CMC
for rnk in distinct_ranks:
    prob = sum(1 for v in ranks.values() if v <= rnk) / len(ranks.values()) 
    rates.append((rnk, prob))


print("RATES: ", len(rates), "\n", rates)

x_val = [x[0] for x in rates]
y_val = [x[1] for x in rates]


plt.plot(x_val, y_val, 'ro')
plt.show()