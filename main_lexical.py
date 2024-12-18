import rule_based.rule_based_utils as rule_based_utils
from datasketch import MinHash
from pre_process import gather_data
from legacy.train import get_counts, get_edges
import copy
from legacy.train import ngram_overlap
import eval_utlis

if __name__ == '__main__':

    base_addr = "NEWS-COPY/"
    method = "ngram"
    # train_data = load_pair_news_copy_in_sent_transformer_format(base_addr, set="train")
    # dev_data = load_pair_news_copy_in_sent_transformer_format(base_addr, set="dev")

    cleaned_text, cleaned_ids = gather_data(
        data_file_path='/home/chen_bowen/TextDedup/NEWS-COPY/test_sets/test_inf_data.json')
    ground_truth_path = '/home/chen_bowen/TextDedup/NEWS-COPY/test_sets/full_test_gt.json'
    print("start n-gram loading")
    if method == "hash":
        num_hashes = 10
        data_dict = rule_based_utils.get_ngrams(copy.deepcopy(cleaned_text), copy.deepcopy(cleaned_ids),
                                                n_gram_size=10, concat=True, char=True)
        m1 = MinHash(num_perm=num_hashes)
        counts = get_counts(data_dict, num_hashes, m1)
        edges_list = get_edges(counts, 4)
    elif method == "ngram":
        print("start n-gram building")
        data_dict = rule_based_utils.get_ngrams(copy.deepcopy(cleaned_text), copy.deepcopy(cleaned_ids),
                                                n_gram_size=25, concat=False, char=True)
        edges_list = ngram_overlap(copy.deepcopy(data_dict), overlap=0.4)
    edge_results, cluster_results, full_results = eval_utlis.get_eval_metrics(edges_list, ground_truth_path,
                                                                    cleaned_ids, community_detection=True)
    print(edge_results)
    print(cluster_results)
    print(full_results)



