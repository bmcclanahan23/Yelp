#This script has the following dependencies - igraph, scipy, matplotlib, nltk, scikit-learn 
#This script also assumes that a directory exists named CSVs 
import json
import igraph
from random import shuffle
from scipy.sparse import lil_matrix,csr_matrix,csc_matrix
from matplotlib import pyplot as plt 
from numpy import array, mean, unique,zeros, argmax,argsort, exp,ones,linspace
from haversine import get_distance
from sklearn.manifold import spectral_embedding
from sklearn.utils.graph import graph_laplacian 
from scipy.sparse.linalg import eigsh
from scipy.stats import pearsonr,mannwhitneyu

def make_xyz_data(filename,lat,lon,value):
    with open(filename,'w') as the_file: 
        the_file.write('lat,lon,value\n')
        for index in range(len(lat)): 
            the_file.write('%f,%f,%f\n'%(lat[index],lon[index],value[index]))



city = 'Phoenix' #city of interest in Yelp dataset 



user_reviews = {}
business_id_to_name = {}
business_info = {}
pos_thresh = 4 #threshold used to consider a review positive 
user_to_index = {}
business_to_index = {}
edges = set([])
weights = []
distances = []
count_thresh = 100 #only consider businesses with this many reviews 
edge_distances = []
show_plot = True 

with open('yelp_academic_dataset_business.json','r') as f: 
    for line in f: 
        business = json.loads(line)
        if  business['city'] == city:
            business_info[business['business_id']] = {'count':0,'lat':business['latitude'],'lon':business['longitude']}            
            business_id_to_name[business['business_id']] = business['name']
num_businesses = len(business_info) 
 
with open('yelp_academic_dataset_review.json','r') as f: 
    for line in f: 
        review = json.loads(line)
        if review['business_id'] in business_info:
            if review['stars'] >=pos_thresh:
            #if review['stars'] <=2:            
                if not review[u'user_id'] in user_reviews: 
                    user_reviews[review[u'user_id']] = set([]) 
                business_info[review['business_id']]['count'] += 1
                user_reviews[review[u'user_id']].add(review['business_id'])
                
                
business_list = [bus for bus in business_info.keys() if business_info[bus]['count'] >= count_thresh]
print 'num businesses ', len(business_list)
shuffle(business_list)
                    
for index,business in enumerate(business_list): 
    business_to_index[business] = index

    
adjacency = lil_matrix((num_businesses,num_businesses))


#construct adjacency matrix   
print 'constructing adjacency matrix'   
for user in user_reviews: 
    user_bus_list = list(user_reviews[user])
    user_bus_list.sort()
    for index_1 in range(len(user_bus_list)):
        for index_2 in range(index_1+1,len(user_bus_list)): 
            if user_bus_list[index_1] in business_to_index and user_bus_list[index_2] in business_to_index:            
                mat_ind_1 = business_to_index[user_bus_list[index_1]]
                mat_ind_2 = business_to_index[user_bus_list[index_2]]
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
#get largest connected component 
graph = graph.subgraph(con_comp[argmax([len(comp) for comp in con_comp])])
#make new business list 
edge_indices = list(range(len(graph.es)))
shuffle(edge_indices)
'''
v_file = open('CSVs\%s_vertices.csv'%city,'w')
e_file = open('CSVs\%s_edges.csv'%city,'w')
v_file.write('lat,lon\n')
e_file.write('lat1,lon1,lat2,lon2\n')

for vertex in graph.vs:
    v_file.write('%f,%f\n'%(business_info[vertex['name']]['lat'],business_info[vertex['name']]['lon']))
    
for e_index in edge_indices[:400]:
    vertex1 = graph.vs[graph.es[e_index].source]
    vertex2 = graph.vs[graph.es[e_index].target]
    e_file.write('%f,%f,%f,%f\n'%(business_info[vertex1['name']]['lat'],business_info[vertex1['name']]['lon'],business_info[vertex2['name']]['lat'],business_info[vertex2['name']]['lon']))
v_file.close()
e_file.close()

business_list = [vertex['name'] for vertex in graph.vs]
'''
#remake adjacency matrix with businesses not in the largest component filtered out 
rows = []
cols = []
data = []
#make new business to index 
print 'removing businesses not in largest component '
for index in range(len(graph.vs)):
    neighbors = graph.neighbors(index)
    cols+=neighbors
    rows += [index for x in range(len(neighbors))]
    data += [adjacency[business_to_index[graph.vs[index]['name']],business_to_index[graph.vs[neigh]['name']]] for neigh in neighbors]

adjacency = csr_matrix((data,(rows,cols)))
'''
norm_graph_lap = graph_laplacian(adjacency,normed=True)
evals_small, evecs_small = eigsh(csc_matrix(norm_graph_lap), 30, sigma=0, which='LM')
plt.figure(1)
plt.plot(evals_small)
plt.title('Eigen Values')
plt.xlabel('order')
plt.ylabel('value')
plt.grid()

plt.figure(2)
[embedding,eigens] = spectral_embedding(adjacency,n_components=30)
plt.plot(list(reversed(-eigens)))
plt.title('Eigen Values from spectral_embedding')
plt.xlabel('order')
plt.ylabel('value')
plt.grid()
plt.show()
'''
#make graph weighted 
print 'add weights'          
graph.es["weight"] = 1.0  
for edge in graph.es: 
    weights.append(adjacency[edge.source,edge.target])


print 'compute unweighted pagerank' 
u_pagerank = graph.pagerank(directed=False,weights=list(ones(len(weights))))
    
print 'compute weighted pagerank' 
pagerank = graph.pagerank(directed=False,weights=weights)

print 'compute unweighted degree' 
degree = graph.degree()

print 'compute weighed degree'
strength = graph.strength(weights=weights)

print 'compute average distance between a business and its neighbors' 
neighbors = graph.neighborhood()
for index,bus in enumerate(business_list):
    iter_distances = []
    point1 = array([[business_info[bus]['lat'],business_info[bus]['lon']]])
    for neighbor in neighbors[index]:    
        point2 = array([[business_info[graph.vs[neighbor]['name']]['lat'],business_info[graph.vs[neighbor]['name']]['lon']]])
        dist = get_distance(point1,point2)[0]
        iter_distances.append(dist)
    distances.append(mean(iter_distances))

print 'computing edge distances '     
for edge in graph.es: 
    point1 = array([[business_info[graph.vs[edge.source]['name']]['lat'],business_info[graph.vs[edge.source]['name']]['lon']]])
    point2 = array([[business_info[graph.vs[edge.target]['name']]['lat'],business_info[graph.vs[edge.target]['name']]['lon']]])
    edge_distances.append(get_distance(point1,point2)[0])

p_values =[]
p_u_values = []
d_values  = []
d_u_values = []
lats = []
lons = []
num_reviews = []
largest_point = 100.0

for index,bus in enumerate(business_list): 
    if business_info[bus]['count'] >= count_thresh: 
          p_values.append(pagerank[index])
          p_u_values.append(u_pagerank[index])
          d_values.append(strength[index])
          d_u_values.append(degree[index])
          lats.append(business_info[bus]['lat'])
          lons.append(business_info[bus]['lon'])
          num_reviews.append(business_info[bus]['count'])
#values = array(values)/(float(max(values))/largest_point) 
#u_values = array(u_values)/(float(max(u_values))/largest_point)
lats = array(lats)
lons = array(lons)
p_values = array(p_values)
p_u_values = array(p_u_values)
d_values = array(d_values)
d_u_values = array(d_u_values)
num_reviews = array(num_reviews)
if show_plot: 
    plt.figure(1)         
    order = argsort(p_values)
    plt.scatter(lons[order],lats[order],c=p_values[order],s=75,cmap='hot')
    plt.title('Business Pagerank Weighted')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar()
    make_xyz_data('CSVs/Weighted_PageRank_%s.csv'%city,lats[order],lons[order],p_values[order])
    
    plt.figure(2)
    order = argsort(p_u_values)
    plt.scatter(lons[order],lats[order],c=p_u_values[order],s=75,cmap='hot')
    plt.title('Business Pagerank Unweighted')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar()
    make_xyz_data('CSVs/Unweighted_PageRank_%s.csv'%city,lats[order],lons[order],p_u_values[order])
    
    plt.figure(3)         
    order = argsort(d_values)
    plt.scatter(lons[order],lats[order],c=d_values[order],s=75,cmap='hot')
    plt.title('Business Degree Weighted')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar()
    make_xyz_data('CSVs/Weighted_Degree_%s.csv'%city,lats[order],lons[order],d_values[order])
    
    plt.figure(4)
    order = argsort(d_u_values)
    plt.scatter(lons[order],lats[order],c=d_u_values[order],s=75,cmap='hot')
    plt.title('Business Degree Unweighted')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar()
    make_xyz_data('CSVs/Unweighted_Degree_%s.csv'%city,lats[order],lons[order],d_u_values[order])
    
    
    plt.figure(5)
    plt.scatter(distances,pagerank)
    plt.title('distance vs weighted pagerank')
    plt.xlabel('distance')
    plt.ylabel('pagerank')
    
    plt.figure(6)
    plt.scatter(distances,u_pagerank)
    plt.title('distance vs unweighted pagerank')
    plt.xlabel('distance')
    plt.ylabel('pagerank')
    
    plt.figure(7)
    plt.scatter(distances,strength)
    plt.title('distance vs weighted degree')
    plt.xlabel('distance')
    plt.ylabel('degree')
    
    plt.figure(8)
    plt.scatter(distances,degree)
    plt.title('distance vs degree')
    plt.xlabel('distance')
    plt.ylabel('degree')
    
    plt.figure(9)
    plt.scatter(edge_distances,weights)
    plt.title('Distance between neighbors vs weights')
    plt.xlabel('distance between neighbors')
    plt.ylabel('weights')
    
    plt.figure(10)
    order = argsort(num_reviews)
    plt.scatter(lons[order],lats[order],c=num_reviews[order],s=75,cmap='hot')
    plt.title('Business Number of Reviews')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar()
    
    plt.figure(11)
    plt.scatter(d_u_values,num_reviews)
    plt.title('degree vs number reviews')
    plt.xlabel('degree')
    plt.ylabel('number reviews')
    print 'degree cor ',mannwhitneyu(num_reviews,d_u_values)   
    
    plt.figure(12)
    plt.scatter(d_values,num_reviews)
    plt.title('weighted degree vs number reviews')
    plt.xlabel('weighted degree')
    plt.ylabel('number reviews')
    print 'strength cor ',mannwhitneyu(num_reviews,d_values)
    
    plt.figure(13)
    plt.scatter(exp(p_u_values),num_reviews)
    plt.title('unweighted pagerank vs number reviews')
    plt.xlabel('unweighted pagerank')
    plt.ylabel('number reviews')
    print 'pagerank cor ',mannwhitneyu(num_reviews,p_u_values)
    
    plt.figure(14)
    plt.scatter(exp(p_values),num_reviews)
    plt.title('weighted pagerank vs number reviews')
    plt.xlabel('weighted pagerank')
    plt.ylabel('number reviews')
    print 'pagerank weighted cor ',mannwhitneyu(num_reviews,p_values)
    
    plt.figure(15)
    plt.hist(d_u_values)
    plt.title('Unweighted Degree Histogram')
    
    plt.figure(16)
    plt.hist(d_values)
    plt.title('Weighted Degree Histogram')
    
    plt.figure(17)
    plt.hist(weights,bins=linspace(0,0.15,10))
    plt.title('Edge Weight Histogram '+city)
    
    plt.show()
     
cent_file = open('pagerank.csv','w')
cent_file.write('location,pagerank\n')
for index, pagerank in enumerate(p_values): 
    cent_file.write('%f %f,%f\n'%(lats[index],lons[index],pagerank))
cent_file.close()
 