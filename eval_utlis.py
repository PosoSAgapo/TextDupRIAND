import json
import random

from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

import cluster_utils


def evaluate(pred_edges, gt_edge_path=None, gt_edges=None, print_metrics=True, print_incorrects=False, two_way=True, save_incorrects=False):

    """
    Return F1, recall, precision, from set of predicted edges and gt set
    """

    if not gt_edges and not gt_edge_path:
        raise ValueError("either gt_edge_path or gt_edges must be specified")

    # Prep ground truth
    if not gt_edges:
        with open(gt_edge_path) as f:
            gt_edges = json.load(f)

    set_gt = set(map(tuple, gt_edges))

    # Prep preds
    pred_edges_list = [[edge[0], edge[1]] for edge in pred_edges]
    set_preds = set(map(tuple, pred_edges_list))

    # Metrics
    if two_way:
        tps = len([i for i in set_gt if i in set_preds or (i[1], i[0]) in set_preds])# if any gold pred in pred which how many gold labels are in pred
        fps = len([i for i in set_preds if i not in set_gt and (i[1], i[0]) not in set_gt])# if any pred not in gold which how many pred labels are not in gold
        fns = len([i for i in set_gt if i not in set_preds and (i[1], i[0]) not in set_preds])# if any gold not in pred which
    else:
        tps = len([i for i in set_gt if i in set_preds])
        fps = len([i for i in set_preds if i not in set_gt])
        fns = len([i for i in set_gt if i not in set_preds])

    if tps + fps > 0:
        precision = tps / (tps + fps)
    else:
        precision = 0
    if tps + fns > 0:
        recall = tps / (tps + fns)
    else:
        recall = 0
    if precision + recall > 0:
        f_score = 2 * (precision * recall) / (precision + recall)
    else:
        f_score = 0

    metrics = {"precision": precision, "recall": recall, "f_score": f_score, "tps": tps, "fps": fps, "fns": fns}

    # Look at wrong ones
    if print_incorrects:
        fp_list = [i for i in set_preds if i not in set_gt]
        fn_list = [i for i in set_gt if i not in set_preds]

        print(fn_list)
        print(len(fn_list))
        print(tps, fps, fns)

    if print_metrics:
        print(metrics)

    if save_incorrects:
        fp_list = [i for i in set_preds if i not in set_gt and (i[1], i[0]) not in set_gt]
        fn_list = [i for i in set_gt if i not in set_preds and (i[1], i[0]) not in set_preds]

        print(tps, fps, fns)

        fp_list = random.sample(fp_list, 50)
        fn_list = random.sample(fn_list, 50)

        return fp_list, fn_list

    else:

        return metrics


def cluster_eval(pred_edges, gt_edges, all_ids):

    """
    Return RI, ARI, NMI, AMI, from set of predicted edges and gt set
    """

    pred_clusters = cluster_utils.clusters_from_edges(pred_edges)

    with open(gt_edges) as f:
        gt_edges = json.load(f)

    set_gt = set(map(tuple, gt_edges))
    gt_clusters = cluster_utils.clusters_from_edges(set_gt)

    # get dictionary mapping article to cluster number
    pred_dict = {}
    pred_count = 0
    for cluster in pred_clusters:
        for article in pred_clusters[cluster]:
            pred_dict[article] = pred_count
        pred_count += 1

    gt_dict = {}
    gt_count = 0
    for cluster in gt_clusters:
        for article in gt_clusters[cluster]:
            gt_dict[article] = gt_count
        gt_count += 1

    # fill in clusters with unclustered articles
    full_pred_clusters = []
    full_gt_clusters = []
    for article in all_ids:
        if article in pred_dict:
            full_pred_clusters.append(pred_dict[article])
        else:
            full_pred_clusters.append(pred_count)
            pred_count += 1

        if article in gt_dict:
            full_gt_clusters.append(gt_dict[article])
        else:
            full_gt_clusters.append(gt_count)
            gt_count += 1

    assert len(full_pred_clusters) == len(full_gt_clusters)

    RI = rand_score(full_pred_clusters, full_gt_clusters)
    ARI = adjusted_rand_score(full_pred_clusters, full_gt_clusters)
    NMI = normalized_mutual_info_score(full_pred_clusters, full_gt_clusters)
    AMI = adjusted_mutual_info_score(full_pred_clusters, full_gt_clusters)

    print({"RI": RI, "ARI": ARI, "NMI": NMI, "AMI": AMI})

    return {"RI": RI, "ARI": ARI, "NMI": NMI, "AMI": AMI}

def get_eval_metrics(edges, gt_path, ids, community_detection=False):
    ''' Gets evaluation metrics after clustering. '''

    # Store different metrics for clustering
    edge_metrics = {}
    cluster_metrics = {}
    full_metrics = {}

    # Perform edge-level evaluation
    metrics = evaluate(edges, gt_edge_path=gt_path)
    for value in metrics:
        edge_metrics[value] = metrics[value]

    # Optionally perform community detection
    if community_detection:
        edges = cluster_utils.detect_communities_nx(edges)

    # Impose transitivty for edges
    cluster_dict = cluster_utils.clusters_from_edges(edges)
    edges = cluster_utils.edges_from_clusters(cluster_dict)

    # Perform cluster-level evaluation (Recall, Precision, F1)
    metrics = evaluate(edges, gt_edge_path=gt_path)
    for value in metrics:
        cluster_metrics[value] = metrics[value]

    # Perform cluster-level evaluation (RI, ARI, NMI, AMI)
    metrics = cluster_eval(pred_edges=edges, gt_edges=gt_path, all_ids=ids)
    for value in metrics:
        full_metrics[value] = metrics[value]

    return edge_metrics, cluster_metrics, full_metrics