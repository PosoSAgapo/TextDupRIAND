from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


def find_consecutive_elements(arr):
    # 初始化结果列表
    consecutive_elements = []
    # 初始化第一个连续元素的索引为 None
    start_index = None
    # 遍历数组元素的索引和值
    for i in range(len(arr)):
        if i > 0 and arr[i] == arr[i - 1]:
            # 当前元素和上一个元素相同
            if start_index is None:
                # 这是连续元素的开始，记录起始下标
                start_index = i - 1
        else:
            # 当前元素和上一个元素不同
            if start_index is not None:
                # 我们找到了一个连续元素的序列，记录这段连续元素的结束下标和值
                consecutive_elements.append((start_index, i - 1, int(arr[i - 1])))
                start_index = None  # 重置起始下标

    # 处理数组最后的连续元素（如果有的话）
    if start_index is not None:
        consecutive_elements.append((start_index, len(arr) - 1, int(arr[-1])))

    return consecutive_elements
def mean_pooling(sentence, tokenizer, model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
def embed(sentence, model, tokenizer):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt',
                              return_overflowing_tokens=True)
    model_output = model(**encoded_input)
    token_embeddings = model_output[0]
    input_mask_expanded = encoded_input.data["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    mean_pooled_emb=torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    truncated_sentence=find_consecutive_elements(encoded_input["overflow_to_sample_mapping"])
    averaged_emb_list=[[] for i in range(len(sentence))]
    for idx, trun_tuple in enumerate(truncated_sentence):
        averaged=torch.sum(mean_pooled_emb[trun_tuple[0]:trun_tuple[1]+1], dim=0)/(trun_tuple[1]-trun_tuple[0]+1)
        averaged_emb_list[trun_tuple[2]]=averaged#averaged sentence over max length
    counter = 0
    continous_counter = 0
    for idx, emb_idx in enumerate(range(len(mean_pooled_emb))):
        if idx not in range(truncated_sentence[continous_counter][0], truncated_sentence[continous_counter][1]+1):
            averaged_emb_list[counter] = mean_pooled_emb[emb_idx]
            counter += 1
        elif idx == truncated_sentence[continous_counter][1]:
            counter += 1
            if continous_counter < len(truncated_sentence)-1:
                continous_counter += 1
    embed = torch.stack(averaged_emb_list, dim=0)
    return embed

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

sentences = ['This is an example sentence', "word "*1000, "word "*1000, 'Each sentence is converted', "word "*1000,  'Each sentence is converted']

with torch.no_grad():
    emb=embed(sentences, model, tokenizer)
    model_output = model(**encoded_input)


