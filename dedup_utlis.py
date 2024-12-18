import faiss
import numpy as np
import pickle

import pandas as pd
from tqdm import tqdm
import math
from glob import glob
import cluster_utils
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import dask.array as da
from dask.distributed import Client, progress
import dask_cudf
from dask_cuda import LocalCUDACluster
import cugraph.dask as dask_cugraph
import cugraph.dask.comms.comms as Comms
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import gc
import pdb
import cugraph as cnx
from datetime import datetime
import torch
import os

def embed(data, model, args):
    all_embeddings = []
    os.makedirs(f"embeddings/{args.dataset}/{args.set}/{args.portion}", exist_ok=True)
    for index in tqdm(range(args.shard_num)):
        if args.recomputing_embedding == True:
            sharded = data.shard(num_shards=args.shard_num, index=index, contiguous=True)
            if args.multi_gpu_paralla_embedding == False:
                corpus_embeddings = model.encode(sharded["text"], batch_size=args.encoding_batch_size,
                                                 show_progress_bar=True)
            else:
                pool = model.start_multi_process_pool()
                corpus_embeddings = model.encode_multi_process(sharded["text"], pool)
                model.stop_multi_process_pool(pool)
            corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
            with open(f"embeddings/{args.dataset}/{args.set}/{args.portion}/{index}.pkl", "wb") as f:
                pickle.dump(corpus_embeddings, f)
        else:
            with open(f"embeddings/{args.dataset}/{args.set}/{args.portion}/{index}.pkl", "rb") as f:
                corpus_embeddings = pickle.load(f)
        all_embeddings.append(corpus_embeddings)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings

def clustering(args, corpus_embedding):
    n_query_batches = math.ceil(corpus_embedding.shape[0] / args.query_batch_size)
    n_corpus_batches = math.ceil(corpus_embedding.shape[0] / args.corpus_batch_size)
    index = faiss.IndexFlatIP(args.embedding_dim)
    if args.multi_gpu_paralla_clustering:
        gpu_resources = []
        for i in range(8):
             res = faiss.StandardGpuResources()
             gpu_resources.append(res)
        #index = faiss.index_factory(args.embedding_dim, "PCA256,IVF2048,Flat")
        #param = 'PCA128,IVF128,PQ16'
        param = 'IVF768,PQ16'
        index = faiss.index_factory(args.embedding_dim, param, faiss.METRIC_INNER_PRODUCT)
        res = [faiss.StandardGpuResources() for i in range(8)]
        #res.setTempMemory(0)
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        for i in range(0, 8):
             vdev.push_back(i)
             vres.push_back(gpu_resources[i])
        co = faiss.GpuMultipleClonerOptions()
        # co.useFloat16 = True
    else:
        res = faiss.StandardGpuResources()
    os.makedirs(f"similarity_results/{args.dataset}/{args.set}/{args.portion}", exist_ok=True)
    for j in tqdm(range(n_corpus_batches)):
        print(f"***** Corpus batch {j} *****")
        if args.multi_gpu_paralla_clustering:
            #gpu_index_flat = faiss.index_cpu_to_gpu_multiple_py(res, index)
            #gpu_index_flat = faiss.index_cpu_to_gpu_multiple(res, index)
            gpu_index_flat = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
            gpu_index_flat.train(corpus_embedding[(args.corpus_batch_size * j):args.corpus_batch_size * (j + 1)])
            gpu_index_flat.nprobe=20
            #gpu_index_flat.train(corpus_embedding[(args.corpus_batch_size * j):args.corpus_batch_size * (j + 1)])
            #gpu_index_flat = faiss.index_cpu_to_all_gpus(cpu_index)
        else:
            gpu_index_flat = faiss.GpuIndexFlatIP(res, args.embedding_dim)
        gpu_index_flat.add(faiss.normalize_L2(corpus_embedding[(args.corpus_batch_size * j):args.corpus_batch_size * (j + 1)]))
        for i in tqdm(range(n_query_batches)):
            D, I = gpu_index_flat.search(faiss.normalize_L2(corpus_embedding[i * args.query_batch_size:(i + 1) * args.query_batch_size]),
                                         args.neighbour_size)
            pdb.set_trace()
            I_adj = I + j * args.corpus_batch_size
            np.save(f"similarity_results/{args.dataset}/{args.set}/{args.portion}/nn_list_batch_{i}_{j}.npy", I_adj)
            np.save(f"similarity_results/{args.dataset}/{args.set}/{args.portion}/dist_list_batch_{i}_{j}.npy", D)
        gpu_index_flat.reset()

def build_graph_nras(args, threshold, edge_df):
    G = cnx.Graph()
    G.from_cudf_edgelist(edge_df, source='src', destination='dst')
    edges, ccs_pd, ccs = cluster_utils.gpu_connected_components(G, args,
                                                                save_file=f'similarity_results/{args.dataset}/{args.set}/{args.portion}/edges/community_edges_{threshold}.pkl',
                                                                detect_communities=True)
    return edges, ccs_pd, ccs

def build_graph_clnda(args, threshold, data):

    chunksize = dask_cugraph.get_chunksize(f"similarity_results/{args.dataset}/{args.set}/{args.portion}/nn_edge_list_{threshold}.csv")
    print(f"obtain chunksize:{chunksize}")
    time_graph_start = datetime.now()
    edge_df = dask_cudf.read_csv(f'similarity_results/{args.dataset}/{args.set}/{args.portion}/nn_edge_list_{threshold}.csv',
                             chunksize=chunksize)
    #edge_df = pd.read_csv(f'similarity_results/{args.dataset}/{args.set}/nn_edge_list_{threshold}.csv')
    edge_df = edge_df.astype("int32")
    G = cnx.Graph()
    G.from_dask_cudf_edgelist(edge_df, source='src', destination='dst')
    #G.from_pandas_edgelist(edge_df, source='src', destination='dst')
    time_graph_end = datetime.now()
    print("Time taken to make graph: ", time_graph_end - time_graph_start)
    edges, ccs_pd, ccs = cluster_utils.gpu_connected_components(G, args,
                                                                save_file=f'similarity_results/{args.dataset}/{args.set}/{args.portion}/edges/community_edges_{threshold}.pkl',
                                                                detect_communities=True)
    return edges, ccs_pd, ccs

def filter_row(nn_row, dist_row, threshold):
    temp_th ={}
    for threshold in threshold:
        temp_th[threshold] = []
    for thresh in threshold:
        temp_th[thresh].append(nn_row[dist_row >= thresh])
    """处理单行的过滤函数"""
    return temp_th

def filtering(args):
    if args.filtering:
        nn_files = glob(f"similarity_results/{args.dataset}/{args.set}/{args.portion}/dist_list_batch*.npy")
        i_list = set([int(file.split("_")[-2]) for file in nn_files])
        j_list = set([int(file.split("_")[-1].split(".")[0]) for file in nn_files])
        i_list = sorted(i_list)
        j_list = sorted(j_list)
        if args.comparision:
            for threshold in args.threshold:
                under_th = []
                for i in tqdm(i_list):  # For all batches of queries
                    dist_list = []
                    nn_list = []
                    for j in j_list:  # Grab results for all batches of the corpus
                        dist_list.append(np.load(f"similarity_results/{args.dataset}/{args.set}/{args.portion}/dist_list_batch_{i}_{j}.npy"))
                        nn_list.append(np.load(f"similarity_results/{args.dataset}/{args.set}/{args.portion}/nn_list_batch_{i}_{j}.npy"))
                    dist_list = np.concatenate(dist_list, axis=1)
                    nn_list = np.concatenate(nn_list, axis=1)
                    under_th.extend([nn_list[i][(dist_list[i] >= threshold)] for i in range(len(nn_list))])
                with open(f'similarity_results/{args.dataset}/{args.set}/under_th_{threshold}.pkl', 'wb') as f:
                    pickle.dump(under_th, f, protocol=4)
                print(len(under_th))
        else:
            under_allth = {}
            counter = 0
            for threshold in args.threshold:
                under_allth[threshold] = []
            for i in tqdm(i_list):  # For all batches of queries
                print(f"***** Query batch {i} *****")
                temp_allth = {}
                for threshold in args.threshold:
                    temp_allth[threshold] = []
                for j in [0, 1]:  # Grab results for all batches of the corpus
                    if j == 0:
                        start = 0
                        end = int(len(j_list) / 2)
                        if start == end:
                            continue
                    else:
                        start = int(len(j_list) / 2)
                        end = len(j_list)
                    print(f"Batched Corpus {j}")
                    dist_list = []
                    nn_list = []
                    for temp in range(start, end):
                        print(f"***** Corpus batch {temp} *****")
                        temp_dist = np.load(f"similarity_results/{args.dataset}/{args.set}/{args.portion}/dist_list_batch_{i}_{temp}.npy", mmap_mode="r")
                        temp_nn = np.load(f"similarity_results/{args.dataset}/{args.set}/{args.portion}/nn_list_batch_{i}_{temp}.npy", mmap_mode="r")
                        dist_list.append(temp_dist)
                        nn_list.append(temp_nn)
                    dist_list = da.concatenate(dist_list, axis=1)
                    nn_list = da.concatenate(nn_list, axis=1)
                    dist_list = dist_list.compute()
                    nn_list = nn_list.compute()
                    num_gpus = torch.cuda.device_count()
                    print("loaded")
                    while True:
                        try:
                            rows_per_segment = dist_list.shape[0] // num_gpus
                            segment_results = {thresh: [None] * num_gpus for thresh in args.threshold}
                            for gpu_idx in range(num_gpus):
                                start_idx = gpu_idx * rows_per_segment
                                end_idx = (gpu_idx + 1) * rows_per_segment if gpu_idx != num_gpus - 1 else dist_list.shape[0]
                                # 分割NumPy矩阵并转换为PyTorch张量
                                dist_segment = torch.from_numpy(dist_list[start_idx:end_idx]).cuda(0)
                                nn_segment = torch.from_numpy(nn_list[start_idx:end_idx]).cuda(0)
                                for thresh in args.threshold:
                                    print(f"Processing batch {gpu_idx} for threshold: ", thresh)
                                    result_segment = nn_segment.masked_fill(~(dist_segment > thresh), -1)
                                    segment_results[thresh][gpu_idx] = result_segment.cpu()
                                    del result_segment
                                    torch.cuda.empty_cache()
                                del dist_segment, nn_segment
                                torch.cuda.empty_cache()
                            break
                        except torch.cuda.OutOfMemoryError:
                            print("Cuda Memo Error")
                            num_gpus *= 2
                    for thresh in args.threshold:
                        # 将所有分割结果汇总到一个列表中
                        if j==0:
                            temp_allth[thresh] = torch.cat(segment_results[thresh], dim=0)
                        else:
                            if 0 == int(len(j_list) / 2):
                                temp_allth[thresh] = torch.cat(segment_results[thresh], dim=0)
                            else:
                                temp=torch.cat(segment_results[thresh], dim=0)
                                temp_allth[thresh] = torch.cat((temp_allth[thresh], temp), dim=1)
                with open(f"similarity_results/{args.dataset}/{args.set}/{args.portion}/under_th_{threshold}_{counter}.pkl", "wb") as f:
                    pickle.dump(temp_allth[thresh], f, protocol=4)
                counter += 1
            del dist_list, nn_list
            del under_allth
            gc.collect()

def edge_building(args, threshold):
    if args.comparision:
        with open(f'similarity_results/{args.dataset}/{args.set}/under_th_{threshold}.pkl', 'rb') as f:
            under_th = pickle.load(f)
        print("Loading filtered edge data finished")
        print("Sanity check ...")
        for i in tqdm(range(len(under_th))):
            assert i in under_th[i]
        print("Creating pairs in NRaS...")
        edge_list_all = [(i, j) for i in range(len(under_th)) for j in under_th[i] if j != i]
        # Remove edges that are in twice
        print("Removing duplicate edges in NRaS ...")
        edge_list_all = list({*map(tuple, map(sorted, edge_list_all))})
    else:
        under_th_files = glob(f"similarity_results/{args.dataset}/{args.set}/{args.portion}/under_th_{threshold}_*.pkl")
        under_th_indexes = set([int(file.split("_")[-1].split(".")[0]) for file in under_th_files])
        edge_set_all = set()
        for under_th_idx in tqdm(under_th_indexes):
            with open(
                    f'similarity_results/{args.dataset}/{args.set}/{args.portion}/under_th_{threshold}_{under_th_idx}.pkl',
                    'rb') as f:
                under_th = pickle.load(f)
            print(f"Loading filtered edge data finished")
            print(f"Creating pairs for underth index {under_th_idx}...")
            # final_edge_list = parallel_process_edges(under_th, args.num_threads)
            num_gpus = torch.cuda.device_count()
            while True:
                try:
                    edge_set = set()
                    rows_per_segment = under_th.shape[0] // num_gpus
                    for gpu_idx in tqdm(range(num_gpus)):
                        start_idx = gpu_idx * rows_per_segment
                        end_idx = (gpu_idx + 1) * rows_per_segment if gpu_idx != num_gpus - 1 else under_th.shape[0]
                        dist_segment = under_th[start_idx:end_idx].cuda(0)
                        # 分割NumPy矩阵并转换为PyTorch张量
                        for i in tqdm(range(len(dist_segment))):
                            process_row(i, dist_segment[i], edge_set)
                        del dist_segment
                        torch.cuda.empty_cache()
                    break
                except torch.cuda.OutOfMemoryError:
                    print("Memory Error")
                    start = datetime.datetime.now()
                    num_gpus *= 2
            edge_set_all = edge_set_all | edge_set
        edge_list_all = list(edge_set_all)
    print("Total edges:", len(edge_list_all))
    return edge_list_all

def deduplicaiton(args, ccs_pd):
    if args.comparision:
        index_to_remove = []
        unique_labels = ccs_pd.labels.unique()
        for label in tqdm(unique_labels):
            sub_frame = ccs_pd[ccs_pd["labels"] == label]
            sub_frame_list = sub_frame["vertex"].tolist()
            index_to_remove.extend(sub_frame_list[1:])
    else:
        unique_labels = ccs_pd.labels.unique()
        index_to_remove = []
        all_label = torch.from_numpy(ccs_pd["labels"].to_numpy()).cuda(0)
        all_vertex = torch.from_numpy(ccs_pd["vertex"].to_numpy()).cuda(0)
        for label in tqdm(unique_labels):
            index_to_remove.extend(remove_indices_for_label(ccs_pd,all_label, all_vertex, label))
    return index_to_remove
def process_row(i, under_th_i, edge_set):
    under_th_i[under_th_i == i] = -1
    selected = under_th_i[under_th_i != -1]
    selected = selected.cpu()
    pdb.set_trace()
    for j in selected:
        if j > i:
            edge = tuple((i,j))
        else:
            edge = tuple((j,i))
        edge_set.add(edge)
    del selected

def parallel_process_edges(under_th, num_threads):
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务到进程池
        futures = {executor.submit(process_row, i, under_th[i]): i for i in range(len(under_th))}

        # 初始化每个行索引对应的结果列表
        ordered_results = [None] * len(under_th)

        # 使用进度条显示处理进度
        with tqdm(total=len(futures)) as progress_bar:
            for future in concurrent.futures.as_completed(futures):
                row_idx = futures[future]  # 获取行索引
                result = future.result()  # 获取结果
                ordered_results[row_idx] = result  # 保存结果到正确的位置，保持顺序
                progress_bar.update(1)  # 更新进度条

    # 将所有结果合并，保持行顺序
    edge_list = [item for sublist in ordered_results if sublist is not None for item in sublist]
    return edge_list


def unique_and_sort_edges(sublist):
    # 将每个元组排序，并转换成集合去除重复元素
    unique_sorted_edges = list({*map(tuple, map(sorted, sublist))})
    return unique_sorted_edges


# 定义主函数，该函数并行化上述去重和排序操作
def parallel_unique_and_sort(edges, num_workers):
    # 获取每个工作线程/进程的工作量大小
    chunk_size = (len(edges) + num_workers - 1) // num_workers

    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 为每个工作线程/进程分配工作
        futures = [executor.submit(unique_and_sort_edges, edges[i:i + chunk_size])
                   for i in range(0, len(edges), chunk_size)]

        # 等待所有工作完成，并收集结果
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

    # 对最终结果再次去重，因为不同的子列表间可能存在重复
    final_edge_list = list(set(map(tuple, results)))

    # 返回最终去重排序后的边列表
    return final_edge_list

def remove_indices_for_label(ccs_pd,all_label, all_vertex, label):
    sub_frame_list = all_vertex[all_label==label].cpu().tolist()
    return sub_frame_list[1:]


