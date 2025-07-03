import numpy as np
import glob
from data_settings import data_path

model = 'drone'
dist = []
for item in glob.glob(data_path[model]+'ground_truth_npy/*'):
    dist.append(np.load(item).sum())
bins = int(2 * len(dist) ** (1/3))
print(bins)
counts, bin_edges = np.histogram(dist,bins)
print(list(counts),list(bin_edges))
threshold = 0.05 * len(dist)
outlier_bins = bin_edges[:-1][counts < threshold]
outlier_bins = [(i,bin_edges[bin_edges.tolist().index(i)+1]) for i in outlier_bins]
print(outlier_bins)
print(np.array(dist).mean())
print(np.array(dist).std())
print(len(dist))

with open(data_path[model]+'outliers/'+model+'_outlier.npy','wb') as f:
    np.save(f, outlier_bins)

