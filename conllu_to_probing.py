from collections import defaultdict
from nltk.tokenize import wordpunct_tokenize
import numpy as np
import random
import csv
from conllu import parse_tree


# Находим токен, у которого есть заданная категория и который находится выше всего в дереве.
def find_category(category, head, children):
    if head['feats']:
        if category in head['feats']:
            return head
    for token in children:
        token_info = token.token
        result = find_category(category, token_info, token.children)
        if result:
            return result


# Классифицируем предложения по значениям выбранной категории. probing_data - {category_value: list of sentences}
def classify(token_trees, category):
    print('Классифицируем предложения по категории...')
    probing_data = defaultdict(list)
    for token_tree in token_trees:
        s_text = ' '.join(wordpunct_tokenize(token_tree.metadata['text']))
        root = token_tree.token
        category_token = find_category(category, root, token_tree.children)
        if category_token:
            value = category_token['feats'][category]
            probing_data[value].append(s_text)
    return probing_data


# Определяем размер выборки gram value с самым маленьким количеством примеров
def limit_size(classified_sentences):
    rare_value = min(classified_sentences, key=lambda i: len(classified_sentences[i]))
    return len(classified_sentences[rare_value])


# Делим на 3 выборки в заданном соотношении
def subsamples_split(x, partition, shuffle):
    print('Разбиваем на выборки...')
    partition_sets = {
        'tr': {},
        'va': {},
        'te': {},
    }
    partition = np.array(partition)
    if np.sum(partition) != 1:
        raise ValueError('the sum of the parts must equal 1')
    if shuffle:
        for sentences in x.values():
            random.shuffle(sentences)
    size = limit_size(x)
    partition=np.cumsum(np.floor(partition * size)).astype(int)
    prev_part = 0
    for size, part in zip(partition, partition_sets):
        values = list(x.keys())
        for value in values:
            partition_sets[part][value] = x[value][prev_part:size]
        prev_part = size
    return partition_sets


# запись в файл в формате пробинга
def writer(result_path, partition_sets):
    print('Записываем в файл...')
    with open(result_path, 'w', encoding='utf-8') as newf:
        my_writer = csv.writer(newf, delimiter='\t', lineterminator='\n')
        for part in partition_sets:
            for value in partition_sets[part]:
                sentences = partition_sets[part][value]
                for sentence in sentences:
                    my_writer.writerow([part, value, sentence])


# гененируем файл для категории из conllu
def generate_probing_file(category, conllu_path, result_path, partition=(0.8, 0.1, 0.1), shuffle=True):
    print('Начинаем генерировать файл для категориии', category, '...')
    with open(conllu_path, 'r', encoding='utf-8') as f:
        conllufile = f.read()
    sentences = parse_tree(conllufile)
    classified_sentences = classify(sentences, category)
    if len(classified_sentences) == 0:
        raise ValueError('It seems like there is no', category, 'category in this language.')
    parts = subsamples_split(classified_sentences, partition, shuffle)
    writer(result_path, parts)


