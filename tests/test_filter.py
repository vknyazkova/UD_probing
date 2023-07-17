from itertools import product

from probing.ud_filter.filtering_probing import ProbingConlluFilter
from probing.ud_filter.sentence_filter import SentenceFilter

import pytest
import unittest
from conllu import parse
from test_conllus import text_testfile1 # пока так, но надо записать в все в conllu file
from queries import by_passive, SOmatchingNumber, ADPdistance


@pytest.mark.sentence_filter
class TestSentenceFilter(unittest.TestCase):
    # path_testfile1 = Path(Path(__file__).parent.resolve(), "by_passive.conllu")
    # text_testfile1 = open(path_testfile1, encoding="utf-8").read()
    trees_testfile1 = parse(text_testfile1)

    def test_token_match_node(self):
        sf = SentenceFilter(self.trees_testfile1[0])
        token = self.trees_testfile1[0][5]
        patterns = [
            {'V': {'upos': 'AUX'}},
            {'V': {'upos': 'VERB'}},
            {'V': {'Number': 'Sing', 'Person': '1'}},
            {'V': {'Number': 'Plur', 'Person': '1'}},
            {'V': {'exclude': ['Definite', 'PronType']}},
            {'V': {'exclude': ['PronType', 'Number']}},
            {'V': {'deprels': 'aux'}},
        ]
        results = [sf.token_match_node(token, node_pattern=p['V']) for p in patterns]
        answers = [True, False, True, False, True, False, False]
        self.assertEqual(results, answers)

    def test__search_suitable_tokens(self):
        sent = self.trees_testfile1[0]
        sf = SentenceFilter(sent)
        node_pattern = {
            'N': {'Number': 'Sing'},  # слова в ед.ч.
            'M': {'upos': '^(?!NOUN|PRON$).*$', 'Number': 'Sing'},  # слова в ед.ч, но не местоимения с существительными
            'K': {'Number': 'Sing', 'exclude': ['PronType']}  # слова в ед.ч но без категории 'PronType'
        }
        answers = {
            'N': ['I', 'I', 'be', 'this', 'way', 'staff', 'member', 'club', 'owner'],
            'M': ['be', 'this'],
            'K': ['be', 'way', 'staff', 'member', 'club', 'owner']
        }
        sf.node_pattern = node_pattern
        sf.nodes_tokens = {node: [] for node in sf.node_pattern}
        for node in node_pattern:
            sf._search_suitable_tokens(node)
        results = {n: [sent.filter(id=t + 1)[0]['lemma'] for t in sf.nodes_tokens[n]] for n in sf.nodes_tokens}
        self.assertEqual(results, answers)

    def test__find_all_nodes(self):
        sf = SentenceFilter(self.trees_testfile1[4])

        sf.node_pattern = by_passive[0]
        sf.nodes_tokens = {node: [] for node in sf.node_pattern}
        self.assertEqual(sf._find_all_nodes(), True)

        sf.node_pattern = ADPdistance[0]
        sf.nodes_tokens = {node: [] for node in sf.node_pattern}
        self.assertEqual(sf._find_all_nodes(), True)

        sf = SentenceFilter(self.trees_testfile1[7])

        sf.node_pattern = ADPdistance[0]
        sf.nodes_tokens = {node: [] for node in sf.node_pattern}
        self.assertEqual(sf._find_all_nodes(), False)

        sf.node_pattern = by_passive[0]
        sf.nodes_tokens = {node: [] for node in sf.node_pattern}
        self.assertEqual(sf._find_all_nodes(), False)

    def test__pattern_relations(self):
        sent = self.trees_testfile1[0]
        sf = SentenceFilter(sent)
        sf.sent_deprels = sf.all_deprels(sf.sentence)
        patterns = ['nsubj(:.*)?',  # nsubj, nsubj:pass
                    '.mod(:.*)?',  # amod, nmid:poss
                    'aux']  # aux
        answers = [['nsubj', 'nsubj:pass'],
                   ['nmod:poss', 'amod'],
                   ['aux']]
        for p, a in zip(patterns, answers):
            self.assertEqual(sf._pattern_relations(p), a)

    def test__pairs_matching_relpattern(self):
        sent = self.trees_testfile1[0]
        sf = SentenceFilter(sent)
        relpattern = {
            ('N', 'M'): {'deprels': 'nsubj(:.*)?'}
        }
        answer = {(2, 0), (7, 4)}

        sf.nodes_tokens = {
            'N': [i for i in range(len(sent)) if i != 16],
            'M': [i for i in range(len(sent)) if i != 16]
        }
        sf.possible_token_pairs = {('N', 'M'): list(product(sf.nodes_tokens['N'], sf.nodes_tokens['M']))}
        sf.sent_deprels = sf.all_deprels(sf.sentence)
        sf.constraints = relpattern

        assert answer == sf._pairs_matching_relpattern(('N', 'M'))

    def test__linear_distance(self):
        sent = self.trees_testfile1[0]
        sf = SentenceFilter(sent)
        sf.possible_token_pairs = {('N', 'M'): [(2, 3), (15, 12), (1, 9)]}
        sf.constraints = {
            ('N', 'M'): {'lindist': (-3, 5)}
        }
        answer = {(2, 3), (15, 12)}
        self.assertEqual(sf._linear_distance(('N', 'M')), answer)

    def test__pair_match_fconstraint(self):
        sent = self.trees_testfile1[1]
        sf = SentenceFilter(sent)
        fconstraints = {
            'intersec': ['VerbForm', 'Number'],
            'disjoint': ['Tense']
        }
        token_pairs = [
            (4, 11),  # matches
            (4, 35),  # Same value for Tense, while should be different
            (11, 35),  # Different value for Number, while should be the same
            (35, 36)  # One token doesn't have pne of the categories
        ]
        answers = [True, False, False, False]
        results = [sf._pair_match_fconstraint(tp, fconstraints) for tp in token_pairs]
        self.assertEqual(answers, results)

    def test__feature_constraint(self):
        sent = self.trees_testfile1[1]
        sf = SentenceFilter(sent)
        sf.constraints = {
            ('N', 'M'): {'fconstraint': {
                'intersec': ['VerbForm'],
                'disjoint': ['Tense']
            }
            }
        }

        sf.nodes_tokens = {
            'N': [i for i in range(len(sent))],
            'M': [i for i in range(len(sent))]
        }
        sf.possible_token_pairs = {('N', 'M'): list(product(sf.nodes_tokens['N'], sf.nodes_tokens['M']))}
        answer = {(4, 11), (11, 4), (11, 35), (35, 11)}
        self.assertEqual(answer, sf._feature_constraint(('N', 'M')))

    def test__find_isomorphism(self):
        sf = SentenceFilter(self.trees_testfile1[1])
        sf.possible_token_pairs = {
            ('V', 'S'): {(12, 11), (0, 4)},
            ('V', 'N'): {(12, 23), (36, 39)},
            ('N', 'BY'): {(39, 37)}
        }
        self.assertEqual(sf._find_isomorphism(), False)
        sf.possible_token_pairs = {
            ('V', 'S'): {(12, 11), (36, 35), (0, 4)},
            ('V', 'N'): {(36, 39), (0, 3), (12, 23)},
            ('N', 'BY'): {(39, 37)}
        }
        self.assertEqual(sf._find_isomorphism(), True)

    def test_filter_sentence(self):
        # all tokens are found, but constraints??
        sf = SentenceFilter(self.trees_testfile1[4])
        self.assertEqual(sf.filter_sentence(by_passive[0], by_passive[1]), False)  # no ('V', 'N'): {'deprels': 'obl'}
        self.assertEqual(sf.filter_sentence(SOmatchingNumber[0], SOmatchingNumber[1]), False)  # no ('V', 'O'): {
        # 'deprels': 'obj'}
        self.assertEqual(sf.filter_sentence(ADPdistance[0], ADPdistance[1]), True)  # все прошло

        # all tokens are found, every constraint has a possible pair, but isomorphism??
        sf = SentenceFilter(self.trees_testfile1[3])
        self.assertEqual(sf.filter_sentence(SOmatchingNumber[0], SOmatchingNumber[1]), True)  # все ок
        sf = SentenceFilter(self.trees_testfile1[6])
        self.assertEqual(sf.filter_sentence(SOmatchingNumber[0], SOmatchingNumber[1]), False)  # нашлись все токены,
        # но проверку на граф не прошли

        # some tokens are not found
        sf = SentenceFilter(self.trees_testfile1[7])
        self.assertEqual(sf.filter_sentence(by_passive[0], by_passive[1]), False)
        self.assertEqual(sf.filter_sentence(ADPdistance[0], ADPdistance[1]), False)


@pytest.mark.filtering_probing
class TestProbingConlluFilter(unittest.TestCase):
    # path_testfile1 = Path(Path(__file__).parent.resolve(), "by_passive.conllu")
    # text_testfile1 = open(path_testfile1, encoding="utf-8").read()
    trees_testfile1 = parse(text_testfile1)

    def test__filter_conllu(self):
        probing_filter = ProbingConlluFilter()
        probing_filter.sentences = self.trees_testfile1
        probing_filter.classes = {'by_passive': by_passive,
                                  'SOmatchingNumber': SOmatchingNumber,
                                  'ADPdistance': ADPdistance}

        by_passive_res = ["I would understand if I was being treated this way by a staff member but the club ' s "
                          "actual OWNER ?!",
                          'Attached for your review are copies of the settlement documents that were filed today in '
                          'the Gas Industry Restructuring / Natural Gas Strategy proceeding , including the Motion '
                          'for Approval of the Comprehensive Settlement that is supported by thirty signatories to '
                          'the Comprehensive Settlement , the Comprehensive Settlement document itself , '
                          'and the various appendices to the settlement .?']
        SOmatchingNumber_res = ['They are kind of in rank order but as I stated if I find the piece that I like we '
                                'will purchase it .',
                                'Masha bought a frying pan , and the boys bought vegetables']
        ADPdistance_res = ['This would have to be determined on a case by case basis .']

        self.assertEqual(probing_filter._filter_conllu('by_passive'), by_passive_res)
        self.assertEqual(probing_filter._filter_conllu('SOmatchingNumber'), SOmatchingNumber_res)
        self.assertEqual(probing_filter._filter_conllu('ADPdistance'), ADPdistance_res)

