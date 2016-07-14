#This script has the following dependencies - igraph, scipy, matplotlib, nltk, scikit-learn 
import json
import re 
import igraph
import pickle 
from random import shuffle
from scipy.sparse import lil_matrix,csr_matrix
from matplotlib import pyplot as plt 
from numpy import  argmax,argsort,array,zeros
from sklearn.cluster import SpectralClustering
from nltk import word_tokenize,FreqDist
from nltk.corpus import stopwords
from nltk.stem.porter import *

def make_xyz_data(filename,lat,lon,value):
    with open(filename,'w') as the_file: 
        the_file.write('lat,lon,value\n')
        for index in range(len(lat)): 
            the_file.write('%f,%f,%f\n'%(lat[index],lon[index],value[index]))
    

def make_xyzcat_data(filename,lat,lon,value,cats):
    with open(filename,'w') as the_file: 
        the_file.write('lat,lon,value\n')
        for index in range(len(lat)): 
            the_file.write('%f,%f,%f,%s\n'%(lat[index],lon[index],value[index],','.join(cats[index])))
    

stemmer = PorterStemmer()
city = 'Las Vegas' #city to construct graph over. Change this to the city of interest 



user_reviews = {}
business_reviews = {}
business_review_sets = {}
words = []
business_id_to_name = {}
business_info = {}
pos_thresh = 4 #Threshold rating to consider a review positive 
user_to_index = {}
business_to_index = {}
edges = set([])
weights = []
distances = []
count_thresh = view_thresh = 100 #Threshold for business exclusion. Eclude businesses with less than 100 reviews
all_real_categories = []
cluster_real_categories = []
num_clusters = 15 #Number of clusters to use for spectral clustering 
top_real_categories_num = 25 #number of categories to include in the Category Proportions heat map
#view_thresh = 100
edge_distances = []
show_plot = True 

with open('yelp_academic_dataset_business.json','r') as f: 
    for line in f: 
        business = json.loads(line)
        if  business['city'] == city:# 
            business_info[business['business_id']] = {'count':0,'lat':business['latitude'],'lon':business['longitude'],'categories':business['categories']}            
            business_id_to_name[business['business_id']] = business['name']
num_businesses = len(business_info) 
 
with open('yelp_academic_dataset_review.json','r') as f: 
    for line in f: 
        review = json.loads(line)
        if review['business_id'] in business_info:
            if review['stars'] >=pos_thresh:            
                if not review[u'user_id'] in user_reviews: 
                    user_reviews[review[u'user_id']] = set([]) 
                business_info[review['business_id']]['count'] += 1
                user_reviews[review[u'user_id']].add(review['business_id'])
                
                
business_list = [bus for bus in business_info.keys() if business_info[bus]['count'] >= count_thresh]
shuffle(business_list)


index = 0                 
for index,business in enumerate(business_list): 
    #here only make graph with businesses that exceed threshold. Think about this 
    business_to_index[business] = index
       
    
adjacency = lil_matrix((num_businesses,num_businesses))


#construct adjacency matrix   
print 'constructing adjacency matrix'   
for user in user_reviews: 
    user_bus_list = list(user_reviews[user])
    user_bus_list.sort()
    for index_1 in range(len(user_bus_list)-1):
        for index_2 in range(index_1+1,len(user_bus_list)): 
            #only add connections of businesses with a certain amount of counts
            if user_bus_list[index_1] in business_to_index and user_bus_list[index_2] in business_to_index:
                mat_ind_1 = business_to_index[user_bus_list[index_1]]
                mat_ind_2 = business_to_index[user_bus_list[index_2]]
                #Should remove this if as a sanity check because this should never happen!!!!!!!!!!!!!!!!!!!!!!!!!!
                if mat_ind_1 != mat_ind_2: 
                    increment = (1/float(business_info[user_bus_list[index_1]]['count'])+1/float(business_info[user_bus_list[index_2]]['count']))/2.0
                    edges.add((user_bus_list[index_1],user_bus_list[index_2]))
                    adjacency[mat_ind_1,mat_ind_2] += increment
                    adjacency[mat_ind_2,mat_ind_1] += increment 

print 'make graph'
graph = igraph.Graph() 
graph.add_vertices(business_list)
edge_list = list(edges)
graph.add_edges(edge_list) 

con_comp = graph.components()
print 'number of components ',len(con_comp)
#get largest connected component 
graph = graph.subgraph(con_comp[argmax([len(comp) for comp in con_comp])])
#make new business list 
business_set = set([vertex['name'] for vertex in graph.vs])
print 'size of largest connected component ', len(graph.vs)
print 'num edges ',len(graph.es)

#remaked adjacency matrix with businesses not in the largest component filtered out 
rows = []
cols = []
data = []
#make new business to index 
print 'removing businesses not in largest component '
for index in range(len(graph.vs)):
    neighbors = graph.neighbors(index)
    cols += neighbors
    rows += [index for x in range(len(neighbors))]
    data += [adjacency[business_to_index[graph.vs[index]['name']],business_to_index[graph.vs[neigh]['name']]] for neigh in neighbors]

adjacency = csr_matrix((data,(rows,cols)))

model = SpectralClustering(n_clusters=num_clusters,affinity='precomputed')
model.fit(adjacency)
lats = []
lons = []
clusterings = []
counts = []
b_cat_list = []
for index in range(len(graph.vs)): 
    if business_info[graph.vs[index]['name']]['count'] >= view_thresh: 
        lats.append(business_info[graph.vs[index]['name']]['lat'])
        lons.append(business_info[graph.vs[index]['name']]['lon'])
        clusterings.append(model.labels_[index])
        counts.append(business_info[graph.vs[index]['name']]['count'])
        b_cat_list.append(business_info[graph.vs[index]['name']]['categories'])

make_xyz_data('%s_clusterings.csv'%city,lats,lons,clusterings)
#make_xyzcat_data('%s_clusterings_with_categories.csv'%city,lats,lons,clusterings,b_cat_list)


plt.figure(1)        
plt.scatter(lons,lats,c=clusterings,s=75)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clusterings')
#plt.colorbar()

plt.figure(2)
plt.scatter(lons,lats,c=counts,s=75)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Business Review Counts')
plt.colorbar()

with open('yelp_academic_dataset_review.json','r') as f: 
    for line in f: 
        review = json.loads(line)
        if review['business_id'] in business_set:            
            if not review['business_id'] in business_reviews:
               business_reviews[review['business_id']] = []
            #remove anything that is not alphanumeric and stem 
            t_words  = word_tokenize(re.sub('[^a-zA-Z0-9\s]','',review['text'].lower().strip()))     
            t_words =  [stemmer.stem(w) for w in t_words]            
            words.extend(t_words)
            business_reviews[review['business_id']].extend(t_words)
            
               
word_freq_dist = FreqDist(words)
#select top 200 words as stopwords 
top_200 = [word[0] for word in word_freq_dist.items()[:200]]

all_stop_words = set(stopwords.words('english') + [word[0] for word in word_freq_dist.items()[:200]])

#remove stopwords 
for bus in business_reviews: 
    business_reviews[bus] = [word for word in business_reviews[bus] if not word in all_stop_words]
#make word sets for each business 
for bus in business_reviews:
    business_review_sets[bus] = set(business_reviews[bus])

#the naming of these variables is confusing categories is a dictionary where the word distributions for 
#each cluster or stored
#cats_b_cats is a dictionary that stores the categories which exist for each cluster 
#all_cats_b is a set that contains all observed categories across all clusters   
categories = {}
real_categories = {}
cats_b_cats = {}
all_cats_b = set([])
cat_freq_dist = {}
cat_word_doc_incl = {}
bus_in_clust = {}
contains_cat = 0
#copute average number of occurences of word per review for each business. Then take summation of these averages
#to get an idea of what businesses in the group are about. This way, one business with a lot of reviews doesn't 
#dominate the category(topic or whatever you want to call it). Also record real business categories within each cluster
#(e.g. Restaruants, nightlife)
for index in range(len(graph.vs)): 
    if not model.labels_[index] in categories:
        real_categories[model.labels_[index]] = []
        categories[model.labels_[index]] = {}
        bus_in_clust[model.labels_[index]] = []
        cats_b_cats[model.labels_[index]] = {'cats_list':[],'bus_count':0,'bus_cat_count':0}
    #record real categories
    real_categories[model.labels_[index]].extend(business_info[graph.vs[index]['name']]['categories'])
    all_real_categories.extend(business_info[graph.vs[index]['name']]['categories'])
    bus_in_clust[model.labels_[index]].append(business_id_to_name[graph.vs[index]['name']])
    iter_freq_dist = FreqDist(business_reviews[graph.vs[index]['name']])
    #add categories from the list to the appropriate entry cats_b_cats and also to call_cats_b  
    cats_b_cats[model.labels_[index]]['bus_count']  +=1
    if len(business_info[graph.vs[index]['name']]['categories']) != 0: 
        cats_b_cats[model.labels_[index]]['cats_list'] = cats_b_cats[model.labels_[index]]['cats_list'] + business_info[graph.vs[index]['name']]['categories']
        cats_b_cats[model.labels_[index]]['bus_cat_count'] +=1
        contains_cat +=1 
        all_cats_b = all_cats_b.union(set(business_info[graph.vs[index]['name']]['categories']))
    for word in iter_freq_dist.items():
        if not word[0] in categories[model.labels_[index]]: 
            categories[model.labels_[index]][word[0]] = 0 
        categories[model.labels_[index]][word[0]] +=  word[1]/float(business_info[graph.vs[index]['name']]['count'])

    
topic_file = open('spectral_clustering_topics_%s.csv'%city,'w')
word_incl_file = open('word_inclusion_file_%s.csv'%city,'w')    
bus_file = open('businesses_in_cluster_%s.csv'%city,'w')
top_words = 10
real_cats_array = []
    
#get top real categories
top_real_categories = sorted(FreqDist(all_real_categories).items(),key=lambda x: x[1],reverse=True)[:top_real_categories_num]
top_real_categories_to_index = {item[0]:index for index,item in enumerate(top_real_categories)} 
#find out how many businesses have categories in the top cats_tresh categories 
top_cats_set = set([item[0] for item in top_real_categories])
bus_count = 0 
for index in range(len(graph.vs)): 
    for cat in business_info[graph.vs[index]['name']]['categories']: 
        if cat in top_cats_set: 
            bus_count += 1 
            break
print 'fraction of businesses with categories in top %d categories: %f\n\n'%(top_real_categories_num,bus_count/float(len(graph.vs))) 
   
    
for index,cat in enumerate(categories):
    if index < len(categories)-1:    
        topic_file.write(str(cat)+',')
        word_incl_file.write(str(cat)+',')
    else:
        topic_file.write(str(cat)+'\n')
        word_incl_file.write(str(cat)+'\n')
    cat_freq_dist[cat] = sorted(categories[cat].items(),key=lambda x:x[1],reverse=True)
    #print 'cat ',cat,' ',[word[0] for word in cat_freq_dist[cat][:100]]
    #print 'cat:%d cat_frac:%f b cats '%(cat,cats_b_cats[model.labels_[index]]['bus_cat_count']/float(cats_b_cats[model.labels_[index]]['bus_count'])),FreqDist(cats_b_cats[model.labels_[index]]['cats_list']).items()
    print ' '
    #see what fraction of the the businesses in the cluster have the first word in the 
    #document 
    #get propotions of real categories
    real_cats_vec = zeros(len(top_real_categories_to_index))
    for categ in FreqDist(real_categories[cat]).items(): 
        if categ[0] in top_real_categories_to_index: 
            real_cats_vec[top_real_categories_to_index[categ[0]]] += categ[1]/float(len(bus_in_clust[cat]))
    real_cats_array.append(real_cats_vec)
    cat_word_doc_incl[cat] = []
    for top in range(1,top_words+1):
        count = 0 
        total = 0
        for index in range(len(graph.vs)):
            if model.labels_[index] == cat:
                total +=1
                if cat_freq_dist[cat][top-1][0] in business_review_sets[graph.vs[index]['name']]: 
                    count +=1 
        cat_word_doc_incl[cat].append(count/float(total))
        print 'contain top %d %f: %d/%d'%(top,count/float(total),count,total)

#write stuff to files       
for index in range(top_words): 
    for index2,cat in enumerate(categories):
        if index2 < len(categories)-1:
            topic_file.write(cat_freq_dist[cat][index][0]+',')
            word_incl_file.write(('%.4f'%cat_word_doc_incl[cat][index])+',')
        else: 
            topic_file.write(cat_freq_dist[cat][index][0]+'\n')
            word_incl_file.write(('%.4f'%cat_word_doc_incl[cat][index])+'\n')

#print out businesses in file 
for clust in bus_in_clust:             
    bus_file.write((str(clust)+','+','.join(bus_in_clust[clust])+'\n').encode(errors='ignore'))
bus_file.close()        
topic_file.close()
word_incl_file.close()        
    
pickle.dump(cat_freq_dist,open('no_r_%s_freq_dist.pickle'%city,'wb'))

fig = plt.figure(4,figsize=(10,9))
cat_y_ticks = [cat for cat in categories]
plt.imshow(array(real_cats_array),interpolation='nearest',origin='low')#,extent=[0,len(cat_to_index)-1,0,num_clusters-1])
plt.yticks(range(len(cat_y_ticks)),cat_y_ticks)
plt.xticks(range(len(top_real_categories_to_index)),[item[0] for item in sorted(top_real_categories_to_index.items(),key=lambda x:x[1])],rotation=90,fontsize=12)
plt.title('Category Proportions')                
plt.colorbar(shrink=0.65)
fig.tight_layout()
plt.savefig('%s_Category_Proportions.png'%city)




plt.show() 

print 'cat frac %f, all cats '%(contains_cat/float(len(graph.vs))), len(all_cats_b)
