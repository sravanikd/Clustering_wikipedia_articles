# Clustering_wikipedia_articles

 Clustering Wikipedia articles using PCA & NMF
- firstly created sparse matrix for bag of words brought out from articles by data preprocessing (filtering, stopwords)
- TruncatedSVD is able to perform PCA on sparse arrays in csr_matrix format
- Combined the knowledge of TruncatedSVD and k-means to cluster some popular pages from Wikipedia
- Created a Pipeline object consisting of a TruncatedSVD followed by KMeans
-  Fitted the pipeline to articles, Calculated the cluster labels
- In addition created an NMF instance, fit transform the articles
- created 6 components(clusters by features) and identified top5 words that impact each feature
