business_spectral_clustering.py - Script produces the following files. In each file name below the word
"city" can be replaced with an actual city name (e.g. Las Vegas)
    spectral_clustering_topics_city.csv - gives the top ten words from each cluster 
    from spectral clustering
    word_inclusion_file_city.csv - gives the fraction of businesses in a cluster that 
    the top words of the cluster appear in reviews of
    businesses_in_cluster_city.csv - lists the names of the businesses in each cluster
    city_clusterings.csv - latitude and longitude coordinates of each business within 
    the city along with a cluster assignment for that business after spectral clustering 
    is performed
    
business_graph.py - This script computes the unweighted and weighted degree and pagerank centrality measures for the mutual business customer graph. 
The following files are produced in a folder named CSVs (assumed to exist in the working directory) which contain the latitude and longitude coordinates of each business in the area along with the it's centrality score
    Weighted_PageRank_city.csv
    Unweighted_PageRank_city.csv
    Weighted_Degree_city.csv
    Unweighted_Degree_city.csv
Additionally plots are produced which show the centrality measures plotted against geographic centrality (average distance to other businesses)


