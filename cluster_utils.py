from itertools import combinations
import hdbscan
import networkx as nx
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import networkx.algorithms.community as nx_comm
from datetime import datetime
from tqdm import tqdm
import pickle
# import cugraph as cnx
# import cugraph.dask as dask_cnx
# import cudf as gd
# import dask_cudf
# import dask
# from dask.distributed import Client
# from dask_cuda import LocalCUDACluster
# import cugraph.dask as dask_cugraph
# import cugraph.dask.comms.comms as Comms
# import pandas as pd
# import pickle
# import rmm
import pdb

def cluster(cluster_type, cluster_params, corpus_embeddings, corpus_ids=None):

    """
    Perform specified clustering method
    """
    if cluster_type not in ["agglomerative", "HDBScan", "SLINK"]:
        raise ValueError('cluster_type must be "agglomerative", "HDBScan", "community" or "SLINK"')
    if cluster_type == "agglomerative":
        if "threshold" not in cluster_params:
            raise ValueError('cluster_params must contain "threshold"')
        if "clustering linkage" not in cluster_params:
            raise ValueError('cluster_params must contain "clustering linkage"')
        if "metric" not in cluster_params:
            raise ValueError('cluster_params must contain "metric"')
    if cluster_type == "HDBScan":
        if "min cluster size" not in cluster_params:
            raise ValueError('cluster_params must contain "min cluster size"')
        if "min samples" not in cluster_params:
            raise ValueError('cluster_params must contain "min cluster size"')
    if cluster_type == "SLINK":
        if "min cluster size" not in cluster_params:
            raise ValueError('cluster_params must contain "min cluster size"')
        if "threshold" not in cluster_params:
            raise ValueError('cluster_params must contain "threshold"')
        if "clustering affinity" not in cluster_params:
            raise ValueError('cluster_params must contain "clustering affinity"')

    if cluster_type == "agglomerative":
        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cluster_params["threshold"],
            linkage=cluster_params["clustering linkage"],
            affinity=cluster_params["metric"]
        )

    if cluster_type == "SLINK":
        clustering_model = DBSCAN(
            eps=cluster_params["threshold"],
            min_samples=cluster_params["min cluster size"],
            metric=cluster_params["metric"]
        )

    if cluster_type == "HDBScan":
        clustering_model = hdbscan.HDBSCAN(
            min_cluster_size=cluster_params["min cluster size"],
            min_samples=cluster_params["min samples"],
            gen_min_span_tree=True
        )

    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_ids = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if int(cluster_id) not in clustered_ids:
            clustered_ids[int(cluster_id)] = []

        if corpus_ids:
            clustered_ids[int(cluster_id)].append(corpus_ids[sentence_id])
        else:
            clustered_ids[int(cluster_id)].append(sentence_id)

    # HDBScan has a cluster where it puts all the unassigned nodes
    if cluster_type == "HDBScan" or cluster_type == "SLINK" and -1 in clustered_ids:
        del clustered_ids[-1]

    return clustered_ids


def clusters_from_edges(edges_list):
    """Identify clusters of passages given a dictionary of edges"""

    # clusters via NetworkX
    G = nx.Graph()
    G.add_edges_from(edges_list)
    sub_graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    sub_graph_dict = {}
    for i in range(len(sub_graphs)):
        sub_graph_dict[i] = list(sub_graphs[i].nodes())

    return sub_graph_dict


def edges_from_clusters(cluster_dict):
    """
    Convert every pair in a cluster into an edge
    """
    cluster_edges = []
    for cluster_id in list(cluster_dict.keys()):
        art_ids_list = cluster_dict[cluster_id]
        edge_list = [list(comb) for comb in combinations(art_ids_list, 2)]
        cluster_edges.extend(edge_list)

    return cluster_edges

def detect_communities_nx(edges, resolution=1):

    """Louvain community detection using nx"""

    G = nx.Graph()
    G.add_edges_from(edges)

    communities = nx_comm.louvain_communities(G, resolution=resolution)

    sub_graph_dict = {}
    for i in range(len(communities)):
        sub_graph_dict[i] = list(communities[i])

    return edges_from_clusters(sub_graph_dict)


def cnx_make_graph_from_edges(edge_list):

    """Make a graph from list of lists of neighbors"""
    # cluster = LocalCUDACluster()
    # #rmm.reinitialize(managed_memory=True)
    # client = Client(cluster)
    # Comms.initialize(p2p=True)
    time_graph_start = datetime.now()
    # chunksize = dask_cugraph.get_chunksize("nn_edge_list_0.9.pkl")
    # # Build edges into a gpu dataframe
    # edge_df = gd.DataFrame({'src': gd.Series([i[0] for i in edge_list]), 'dst': gd.Series([i[1] for i in edge_list])})
    # edge_df = dask_cudf.from_cudf(edge_df, npartitions=200, chunksize=chunksize)

    # Make graph
    G = cnx.Graph()
    #G.from_cudf_edgelist(edge_df, source='src', destination='dst')
    G.from_dask_cudf_edgelist(edge_list, source='src', destination='dst')
    #G = nx.Graph()
    #G.from_pandas_edgelist(edge_df, source='src', destination='dst')
    #G = nx.from_pandas_edgelist(edge_df, source='src', target='dst')
    
    #print("Number of nodes:", dask_cnx.structure.mg_property_graph.cugraph.structure.graph_implementation.simpleGraphImpl.number_of_vertices(G))
    #print("Number of edges before imposing transistivty:", dask_cnx.structure.graph_implementation.simpleGraphImpl.number_of_edges(G))

    time_graph_end = datetime.now()
    print("Time taken to make graph: ", time_graph_end-time_graph_start)
    #f = open("similarity_results/lm1b_example/graph.pkl", "wb")
    #pickle.dump(G, f)
    #f.close()
    return G

def gpu_connected_components(G, args, save_file, detect_communities=False):

    """
    Impose transitivity and return edges, either with or without community detection
    """

    time_cc_start = datetime.now()

    print("Imposing transitivity ...")

    if args.comparision:
        if detect_communities:
            ccs, _ = cnx.louvain(G, resolution=1)
            ccs = ccs.rename(columns={"partition": "labels"})
            ccs = ccs.to_pandas()
        else:
            ccs = cnx.connected_components(G)
            ccs = ccs.to_pandas()
    else:
        if detect_communities:
            ccs, _ = dask_cnx.louvain(G, resolution=1)
            ccs = ccs.compute().to_pandas()
            ccs = ccs.rename(columns={"partition": "labels"})
        else:
            ccs = dask_cnx.connected_components(G)
            ccs = ccs.compute().to_pandas()
    print("Distinct connected components: ", ccs.labels.nunique())

    total_perms = 0
    total_reduced = 0

    def get_edges(df_1, df_2, total_perms, total_reduced):
        df_1 = df_1.merge(df_2, on='labels', how='inner')
        df_1 = df_1.drop(['labels'], axis=1)
        total_perms += len(df_1)
        df_1 = df_1[df_1['vertex_x'] < df_1['vertex_y']]  # remove both directions and loops
        total_reduced += len(df_1)
        return df_1, total_perms, total_reduced

    lengths = []
    all_edges = []
    ccs_pd = ccs
    #pdb.set_trace()
    if args.collecting_frame:
        for label in tqdm(ccs_pd.labels.unique()):
            sub_frame = ccs[ccs['labels'] == label]
            lengths.append(len(sub_frame))
            if len(sub_frame) < 50000:  # Larger subframes don't fit on GPU, so run on CPU (though slower!)
                edge_df, total_perms, total_reduced = get_edges(sub_frame, sub_frame, total_perms, total_reduced)
                all_edges.append(edge_df)
            else:
                sub_frame_A = sub_frame[:30000]
                sub_frame_B = sub_frame[30000:]
                edge_df, total_perms, total_reduced = get_edges(sub_frame_A, sub_frame_A, total_perms, total_reduced)
                all_edges.append(edge_df)
                edge_df, total_perms, total_reduced = get_edges(sub_frame_A, sub_frame_B, total_perms, total_reduced)
                all_edges.append(edge_df)
                edge_df, total_perms, total_reduced = get_edges(sub_frame_B, sub_frame_A, total_perms, total_reduced)
                all_edges.append(edge_df)
                edge_df, total_perms, total_reduced = get_edges(sub_frame_B, sub_frame_B, total_perms, total_reduced)
                all_edges.append(edge_df)


        squares = [i*i for i in lengths]

        assert total_perms == sum(squares)
        assert total_reduced == (sum(squares) - len(ccs))/2

        time_cc_end = datetime.now()

        edges = pd.concat(all_edges)
        #edges = gd.concat([gd.DataFrame.from_pandas(x) for x in all_edges])
        assert len(edges) == total_reduced

        print("Time taken to find connected components: ", time_cc_end-time_cc_start)
        print("Number of edges after imposing transitivity:", len(edges))

        with open(save_file, 'wb') as f:
            pickle.dump(edges, f)
        with open(save_file.replace("community_edges.pkl","community_graph.pkl"), 'wb') as f:
            pickle.dump(ccs_pd, f)
        return edges, ccs_pd, ccs
    else:
        return [], ccs_pd, ccs

