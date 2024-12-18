import random

def random_bool(probability=0.5):
    """Returns True with given probability

    Args:
        probability: probability to return True

    """
    assert (0 <= probability <= 1), "probability needs to be >= 0 and <= 1"
    return random.random() < probability

def delete_random_token(line, probability):
    """Delete random tokens in a given String with given probability

    Args:
        line: a String
        probability: probability to delete each token

    """
    line_split = line.split()
    ret = [token for token in line_split if not random_bool(probability)]
    return " ".join(ret)


def replace_random_token(line, probability, filler_token=""):
    """Replace random tokens in a String by a filler token with given probability

    Args:
        line: a String
        probability: probability to replace each token
        filler_token: token replacing chosen tokens

    """
    line_split = line.split()
    for i in range(len(line_split)):
        if random_bool(probability):
            length = len(line_split[i])
            replaced_char_idx = random.sample(range(length), 1)
            if line_split[i][replaced_char_idx[0]].lower() == line_split[i][replaced_char_idx[0]]:
                char_list = [ch for ch in line_split[i]]
                char_list[replaced_char_idx[0]] = random.sample("abcdefghijklmnopqrstuvwxyz", 1)[0]
                line_split[i] = "".join(char_list)
            else:
                char_list = [ch for ch in line_split[i]]
                char_list[replaced_char_idx[0]] = random.sample("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 1)[0]
                line_split[i] = "".join(char_list)
    return " ".join(line_split)

def insert_random_token(line, probability):
    line_split = line.split()
    for i in range(len(line_split)):
        if random_bool(probability):
            length = len(line_split[i])
            replaced_char_idx = random.sample(range(length), 1)
            char_list = [ch for ch in line_split[i]]
            char_list.insert(replaced_char_idx[0], random.sample("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", 1)[0])
            line_split[i] = "".join(char_list)
    return " ".join(line_split)

def insert_random_linefill(line, probability):
    line_split = line.split()
    for i in range(len(line_split)):
        if random_bool(probability):
            length = len(line_split[i])
            replaced_char_idx = random.sample(range(length), 1)
            char_list = [ch for ch in line_split[i]]
            char_list.insert(replaced_char_idx[0], "\n")
            line_split[i] = "".join(char_list)
    return " ".join(line_split)

def swap_random_token(line, probability):
    line_split = line.split()
    for i in range(len(line_split)):
        if random_bool(probability):
            length = len(line_split[i])
            if len(line_split[i]) > 2:
                replaced_char_idx = random.sample(range(length), 2)
                char_list = [ch for ch in line_split[i]]
                temp = char_list[replaced_char_idx[0]]
                char_list[replaced_char_idx[0]] = char_list[replaced_char_idx[1]]
                char_list[replaced_char_idx[1]] = temp
                line_split[i] = "".join(char_list)
    return " ".join(line_split)

def repeat_random_token(line, probability):
    line_split = line.split()
    for i in range(len(line_split)):
        if random_bool(probability):
            length = len(line_split[i])
            replaced_char_idx = random.sample(range(length), 1)
            char_list = [ch for ch in line_split[i]]
            char_list.insert(replaced_char_idx[0], char_list[replaced_char_idx[0]])
            line_split[i] = "".join(char_list)
    return " ".join(line_split)

def random_token_permutation(line, _range):
    """Random permutation over the tokens of a String, restricted to a range, drawn from the uniform distribution

    Args:
        line: a String
        _range: Max range for token permutation

    """
    line_split = line.split()
    new_indices = [i+random.uniform(0, _range+1) for i in range(len(line_split))]
    res = [x for _, x in sorted(zip(new_indices, line_split), key=lambda pair: pair[0])]
    return " ".join(res)

def noise_pipe(raw_text, delete_prob=0.04, replace_prob=0.03, insert_prob=0.01, swap_prob=0.01, repeat_prob=0.04, insert_linefill_prob=0.03, permutation_range=1):
    raw_text = delete_random_token(raw_text, delete_prob)
    raw_text = replace_random_token(raw_text, replace_prob)
    raw_text = insert_random_token(raw_text, insert_prob)
    raw_text = swap_random_token(raw_text, swap_prob)
    raw_text = repeat_random_token(raw_text, repeat_prob)
    raw_text = insert_random_linefill(raw_text, insert_linefill_prob)
    raw_text = random_token_permutation(raw_text, permutation_range)
    return raw_text