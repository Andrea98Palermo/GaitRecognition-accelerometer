from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import euclidean_distances
from imblearn.over_sampling import SMOTE
import math
import sys

seed = 511
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

        pca = PCA(n_components=self.q, random_state=seed).fit(X)                        #PCA
        A_q = pca.components_.T                                                         #componenti principali = colonne di A_q

        kmeans = KMeans(n_clusters=self.n_features, random_state=seed).fit(A_q)         #raggruppamento righe di A_q con KMeans
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_                                       #valore medio di ogni cluster

        dists = defaultdict(list)
        for i, c in enumerate(clusters):                                                #scelta vettore più vicino a media per ogni cluster
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
X_train = df[0 : math.ceil(df.shape[0] / 2)]              
X_test = df[math.ceil(df.shape[0] / 2) : ]
y_train = y[0 : math.ceil(df.shape[0] / 2)]
y_test = y[math.ceil(df.shape[0] / 2) : ]
y_test.reset_index(inplace=True, drop=True)

#lista contenente le identità reali(1-153) per ogni istanza nell'insieme di test (da binarizzare in seguito)
y_test_bin = y_test
#lista in cui verranno aggiunte ordinatamente tutte le identità binarizzate per ogni diverso modello
y_test_bin_all = []
#lista in cui verranno aggiunte ordinatamente tutte le probabilità predette da ogni modello
y_scores_all = []

#classificazione soggetto i([1-153])
for i in range(1, 154):
    
    sm = SMOTE(random_state=seed)                                                       
    y_train_bin = [1 if x == i else 0 for x in y_train.tolist()]                        #binarizzazione identità insieme training
    y_test_bin = [1 if x == i else 0 for x in y_test.tolist()]                          #binarizzazione identità insieme test
    X_train_upsampled, y_train_upsampled = sm.fit_resample(X_train, y_train_bin)        #upsampling classe minoritaria

    #pipeline per normalizzazione e creazione modello
    clf = Pipeline([
        ('scaling', StandardScaler()),
        #('classification', AdaBoostClassifier(random_state=seed, n_estimators=500))
        ('classification', LogisticRegression(random_state=seed, max_iter=20000, solver="saga", multi_class="ovr"))        
    ])
    clf.fit(X_train_upsampled, y_train_upsampled)                           #allenamento modello
    clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')     #reazione modello con probabilità calibrate
    clf_isotonic.fit(X_train_upsampled, y_train_upsampled)

    y_scores = clf_isotonic.predict_proba(X_test)               #predizione probabilità stimate
    y_scores_1d = [x[1] for x in y_scores]                      #vengono mantenute esclusivamente le probabilità di identità genuina
    print("ITERAZIONE ", i)

    y_test_bin_all.extend(y_test_bin)       #aggiunta identità binarizzate a insieme finale per successivo calcolo performance
    y_scores_all.extend(y_scores_1d)        #aggiunta probabilità predette a insieme finale per successivo calcolo performance


FARs = []
FRRs = []
GARs = []
HTERs = []

i = 1
threshs = np.linspace(0,1,101)              #insieme inziale soglie: [0-1] con campionamento 0.01
for t in threshs:
    y_pred_t = [1 if x >= t else 0 for x in y_scores_all]                                           #confronto probabilità/soglie --> produzione predizioni
    tn, fp, fn, tp = metrics.confusion_matrix(y_test_bin_all, y_pred_t, labels=[0,1]).ravel()       #conteggio diverse casistiche
    FAR = fp/(fp + tn)                                                                              #calcolo error rate
    FRR = fn/(fn + tp)
    GAR = 1 - FRR
    HTER = (FAR + FRR)/2 
    FARs = np.append(FARs, FAR)                                                                     #memorizzazione error rate
    FRRs = np.append(FRRs, FRR)
    GARs = np.append(GARs, GAR)
    HTERs = np.append(HTERs, HTER)
    i += 1

#calcolo EER complessivo
abs_diffs = np.abs(FARs - FRRs)             
min_index = np.argmin(abs_diffs)
eer = np.mean((FARs[min_index], FRRs[min_index]))
threshold = threshs[min_index]
print("EER:------------>", eer)
print("treshold:------------>", threshold)

print("FAR:------------>", FARs)
print("FRR:------------>", FRRs)
print("GAR:------------>", GARs)

#calcolo zeroFAR (<0.001)
try:
    zeroFAR_index = np.where(FARs < 0.001)[0][0]
    zeroFAR = FRRs[zeroFAR_index]
    print("ZeroFAR_0.001:------------>", zeroFAR)
except:
    print("Nessun valore FAR < 0.001")

#calcolo zeroFAR (<0.0001)
try:
    zeroFAR_2_index = np.where(FARs < 0.0001)[0][0]
    zeroFAR_2 = FRRs[zeroFAR_2_index]
    print("ZeroFAR_0.0001:------------>", zeroFAR_2)
except:
    print("Nessun valore FAR < 0.0001")





#SECONDO ROUND CON SESSIONI INVERTITE

#divisione sessioni
X_test = df[0 : math.ceil(df.shape[0] / 2)]              
X_train = df[math.ceil(df.shape[0] / 2) : ]
y_test = y[0 : math.ceil(df.shape[0] / 2)]
y_train = y[math.ceil(df.shape[0] / 2) : ]
y_train.reset_index(inplace=True, drop=True)


#lista contenente le identità reali(1-153) per ogni istanza nell'insieme di test (da binarizzare in seguito)
y_test_bin = y_test
#lista in cui verranno aggiunte ordinatamente tutte le identità binarizzate per ogni diverso modello
y_test_bin_all = []
#lista in cui verranno aggiunte ordinatamente tutte le probabilità predette da ogni modello
y_scores_all = []

#classificazione soggetto i([1-153])
for i in range(1, 154):
    
    sm = SMOTE(random_state=seed)                                                       
    y_train_bin = [1 if x == i else 0 for x in y_train.tolist()]                        #binarizzazione identità insieme training
    y_test_bin = [1 if x == i else 0 for x in y_test.tolist()]                          #binarizzazione identità insieme test
    X_train_upsampled, y_train_upsampled = sm.fit_resample(X_train, y_train_bin)        #upsampling classe minoritaria

    #pipeline per normalizzazione e creazione modello
    clf = Pipeline([
        ('scaling', StandardScaler()),
        #('classification', AdaBoostClassifier(random_state=seed, n_estimators=500))
        ('classification', LogisticRegression(random_state=seed, max_iter=20000, solver="saga", multi_class="ovr"))     
    ])
    clf.fit(X_train_upsampled, y_train_upsampled)                           #allenamento modello
    clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')     #creazione modello con probabilità calibrate
    clf_isotonic.fit(X_train_upsampled, y_train_upsampled)

    y_scores = clf_isotonic.predict_proba(X_test)               #predizione probabilità stimate
    y_scores_1d = [x[1] for x in y_scores]                      #vengono mantenute esclusivamente le probabilità di identità genuina
    print("ITERAZIONE ", i)

    y_test_bin_all.extend(y_test_bin)       #aggiunta identità binarizzate a insieme finale per successivo calcolo performance
    y_scores_all.extend(y_scores_1d)        #aggiunta probabilità predette a insieme finale per successivo calcolo performance


FARs = []
FRRs = []
GARs = []
HTERs = []

i = 1
threshs = np.linspace(0,1,101)              #insieme inziale soglie: [0-1] con campionamento 0.01
for t in threshs:
    y_pred_t = [1 if x >= t else 0 for x in y_scores_all]                                           #confronto probabilità/soglie --> produzione predizioni
    tn, fp, fn, tp = metrics.confusion_matrix(y_test_bin_all, y_pred_t, labels=[0,1]).ravel()       #conteggio diverse casistiche
    FAR = fp/(fp + tn)                                                                              #calcolo error rate
    FRR = fn/(fn + tp)
    GAR = 1 - FRR
    HTER = (FAR + FRR)/2 
    FARs = np.append(FARs, FAR)                                                                     #memorizzazione error rate
    FRRs = np.append(FRRs, FRR)
    GARs = np.append(GARs, GAR)
    HTERs = np.append(HTERs, HTER)
    i += 1

#calcolo EER complessivo
abs_diffs = np.abs(FARs - FRRs)             
min_index = np.argmin(abs_diffs)
eer = np.mean((FARs[min_index], FRRs[min_index]))
threshold = threshs[min_index]
print("EER:------------>", eer)
print("treshold:------------>", threshold)

print("FAR:------------>", FARs)
print("FRR:------------>", FRRs)
print("GAR:------------>", GARs)


#calcolo zeroFAR (<0.001)
try:
    zeroFAR_index = np.where(FARs < 0.001)[0][0]
    zeroFAR = FRRs[zeroFAR_index]
    print("ZeroFAR_0.001:------------>", zeroFAR)
except:
    print("Nessun valore FAR < 0.001")

#calcolo zeroFAR (<0.0001)
try:
    zeroFAR_2_index = np.where(FARs < 0.0001)[0][0]
    zeroFAR_2 = FRRs[zeroFAR_2_index]
    print("ZeroFAR_0.0001:------------>", zeroFAR_2)
except:
    print("Nessun valore FAR < 0.0001")
