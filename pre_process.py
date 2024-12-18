import json
import logging
import os

from symspellpy import Verbosity, SymSpell
from tqdm import tqdm
from transformers import BertTokenizerFast

from utils import set_global_logging_level


def spellcheck(list_of_texts, spell_check_type):
    ''' Runs spell-checker over the list of texts. '''

    if spell_check_type == "symspell":
        spell_checked_texts = symspell_check_ocr(list_of_texts)
    if spell_check_type == "fixed":
        spell_checked_texts = fixed_dict(list_of_texts)
    if spell_check_type is None:
        return list_of_texts

    return spell_checked_texts


def symspell_setup(resource_dir="", edit_distance=2):
    sym_spell = SymSpell(max_dictionary_edit_distance=edit_distance, prefix_length=7)

    dictionary_path = os.path.join(resource_dir, "frequency_dictionary_en_82_765.txt")
    bigram_path = os.path.join(resource_dir, "frequency_bigramdictionary_en_243_342.txt")

    print("Dictionary Path:", dictionary_path)
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    return sym_spell


def fixed_dict(ocr_article_clean_texts):
    '''Very flexible spell checker.'''

    sym_spell = symspell_setup(edit_distance=5)

    ocr_spell_texts = []

    print("\n Spell checking ...")
    for text in tqdm(ocr_article_clean_texts):
        spell_corr = []
        for input_term in text.split():
            suggestions = sym_spell.lookup(input_term, Verbosity.TOP, max_edit_distance=None, include_unknown=True,
                                           transfer_casing=True)
            spell_corr.append(suggestions[0].term)
        ocr_spell_texts.append(" ".join(spell_corr))

    return ocr_spell_texts


def symspell_check_ocr(ocr_article_clean_texts):
    """Corrects spelling of OCR article texts"""

    sym_spell = symspell_setup()

    ocr_spell_texts = []

    print("\n Spell checking ...")
    for text in tqdm(ocr_article_clean_texts):
        spell_corr = []
        for input_term in text.split():
            suggestions = sym_spell.lookup(input_term, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True,
                                           transfer_casing=True)
            spell_corr.append(suggestions[0].term)
        ocr_spell_texts.append(" ".join(spell_corr))

    return ocr_spell_texts


def remove_odd_characters(list_of_texts):
    ''' Removes punctuation, unknown characters. '''
    chars_to_remove = r'"#$%&\()*+/:;<=>@[\\]^_`{|}~.?,!\''
    ocr_article_clean_texts = []

    for text in list_of_texts:
        text = text.replace("-\n", "").replace("\n", " ")
        text = text.translate(str.maketrans('', '', chars_to_remove))
        text = text.encode('ascii', 'ignore').decode()
        ocr_article_clean_texts.append(text)

    return ocr_article_clean_texts
def clean_text(corpus_dict, first_n_tok=None, min_tok=None, spell_check="symspell"):
    ''' Cleans texts by removing punctuation, optionally spell-checking. '''

    cleaned_ids = []
    org_texts = []

    # instantiate tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    set_global_logging_level(logging.ERROR, ["transformers", "BertTokenizerFast"])

    for key in list(corpus_dict.keys()):
        text = corpus_dict[key]['byline'] + " " + corpus_dict[key]['article']

        if first_n_tok is not None:
            tokens = tokenizer.encode(text, truncation=False)
            text = tokenizer.decode(tokens[1:first_n_tok])
        if min_tok is not None:
            if len(tokens) > min_tok:
                cleaned_ids.append(key)
                org_texts.append(text)
        else:
            cleaned_ids.append(key)
            org_texts.append(text)

    cleaned_texts = remove_odd_characters(org_texts)

    if spell_check:
        cleaned_texts = spellcheck(cleaned_texts, spell_check_type=spell_check)

    return cleaned_ids, cleaned_texts

def gather_data(data_file_path, min_tok=None, n_tok=None, spell_check=None):
    ''' Gathers article data. '''

    with open(data_file_path) as f:
        corpus_dict = json.load(f)

    cleaned_id_list, cleaned_text_list = clean_text(
        corpus_dict,
        first_n_tok=n_tok,
        min_tok=min_tok,
        spell_check=spell_check
    )
    return cleaned_text_list, cleaned_id_list