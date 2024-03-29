import re
from conllu import models
from itertools import product
from collections import defaultdict
import networkx as nx
import copy


class SentenceFilter:
    def __init__(self, sentence: models.TokenList):
        """
        Checks if a sentence matches patterns

        Attributes:
            sentence: sentence represented as TokenList
            node_pattern: a dictionary with a node_label as a key and a dictionary of feature restrictions as a value.
                sample_node_pattern = {
                    node_label: {
                        field_or_category: pattern,
                        'exclude': [exclude categories]
                        }
                    }
            constraints: a dictionary with a node_pair as a key and a dictionary with different constraints on node_pair
                sample_constraints = {
                    ('N', 'M'): {
                        'deprels': 'some_regexp',
                        'fconstraint': {
                            'disjoint': ['feature1', 'feature2'],
                            'intersec': ['feature1', 'feature2']
                            },
                        'lindist': (1, 10)
                        }
                    }
            sent_deprels: a dictionary of all relations and pairs with these relations {relation: [(head, dependent)]}
            nodes_tokens: a dictionary with all tokens that can be a given node in pattern {node: [token id]}
            possible_token_pairs:
        """

        self.sentence = sentence
        self.node_pattern = None
        self.constraints = None
        self.sent_deprels = {}
        self.nodes_tokens = {}
        self.possible_token_pairs = None

    def token_match_node(self, token: models.Token, node_pattern: dict) -> bool:
        """Checks if a token matches the node_pattern"""

        for feat in node_pattern:
            if feat in token.keys():
                if not re.fullmatch(node_pattern[feat], token[feat], re.I):  # changed from re.match to re.fullmatch
                    return False
            elif token['feats']:
                if feat in token['feats']:
                    if not re.fullmatch(node_pattern[feat], token['feats'][feat],
                                        re.I):  # changed from re.match to re.fullmatch
                        return False
                elif feat == "exclude":
                    for ef in node_pattern[feat]:
                        if ef in token['feats']:
                            return False
                else:
                    return False
            else:
                return False
        return True

    def all_deprels(self, token_list: models.TokenList) -> defaultdict:
        """Returns a dictionary {relation: [(head, dependent)]} of all relations in the sentence"""

        deprels = defaultdict(list)
        for t in token_list:
            if isinstance(t['head'], int) and isinstance(t['id'], int):
                deprels[t['deprel']].append((t['head'] - 1, t['id'] - 1))
        return deprels

    def _search_suitable_tokens(self, node: str):
        """Selects from a token_list those tokens that match given node_pattern
        and saves it in self.nodes_tokens[node]"""

        for token in self.sentence:
            if self.token_match_node(token, self.node_pattern[node]) and isinstance(token['id'], int):
                self.nodes_tokens[node].append(token['id'] - 1)

    def _find_all_nodes(self):
        """Checks if every node in the pattern has at least one matching token"""

        for node in self.node_pattern:
            self._search_suitable_tokens(node)
            if not self.nodes_tokens[node]:
                return False
        return True

    def _pattern_relations(self, rel_pattern: str):
        """Returns all relation names in the sentence that match the given pattern"""

        rels = []
        for rel in self.sent_deprels:
            if re.fullmatch(rel_pattern, rel, re.I):  # changed from re.serach to re.fullmatch
                rels.append(rel)
        return rels

    def _pairs_with_rel(self, node_pair: tuple, rel_name: str) -> set:  # оставить private
        """Returns those pairs of tokens that:
            1) are related by a rel_name
            2) are among possible_token_pairs for a node_pair"""

        if rel_name not in self.sent_deprels:
            return set()
        else:
            return set(self.sent_deprels[rel_name]).intersection(self.possible_token_pairs[node_pair])

    def _pairs_matching_relpattern(self, node_pair: tuple) -> set:
        """Returns a set of token pairs, whose relations match the pattern"""

        all_suitable_rels = set()
        for rel in self._pattern_relations(self.constraints[node_pair]['deprels']):
            all_suitable_rels = all_suitable_rels | self._pairs_with_rel(node_pair, rel)
        return all_suitable_rels

    def _linear_distance(self, node_pair: tuple) -> set:
        """Returns a set of token pairs with a given linear distance between tokens
        :param lindist: tuple(min_distance, max_distance)"""

        suitable_pairs = set()
        lindist = self.constraints[node_pair]['lindist']
        for pair in self.possible_token_pairs[node_pair]:
            dist = pair[1] - pair[0]
            if lindist[0] <= dist <= lindist[1]:
                suitable_pairs.add(pair)
        return suitable_pairs

    def _pair_match_fconstraint(self, token_pair: tuple, fconstraint: dict) -> bool:
        """Checks if a token pair matches all the feature constraints"""

        t1_feats = self.sentence[token_pair[0]]['feats']
        t2_feats = self.sentence[token_pair[1]]['feats']
        if t1_feats and t2_feats:
            for ctype in fconstraint:
                for f in fconstraint[ctype]:
                    if (f in t1_feats) and (f in t2_feats):
                        if ctype == 'intersec':
                            if t1_feats[f] != t2_feats[f]:
                                return False
                        elif ctype == 'disjoint':
                            if t1_feats[f] == t2_feats[f]:
                                return False
                        else:
                            raise ValueError('Wrong feature constraint type')
                    else:
                        return False
            return True
        else:
            return False

    def _feature_constraint(self, node_pair: tuple) -> set:
        """Returns all pairs that match constraints on features"""

        suitable_pairs = set()
        fconstraint = self.constraints[node_pair]['fconstraint']
        for pair in self.possible_token_pairs[node_pair]:
            if self._pair_match_fconstraint(pair, fconstraint):
                suitable_pairs.add(pair)
        return suitable_pairs

    def _find_isomorphism(self):
        """Checks if there is a graph with possible_token_pairs that is isomorphic to a constraint pairs graph"""

        nodes_graph = nx.Graph()
        nodes_graph.add_edges_from(list(self.possible_token_pairs.keys()))
        possible_edges = list(product(*self.possible_token_pairs.values()))
        for edges in possible_edges:
            tokens_graph = nx.Graph()
            tokens_graph.add_edges_from(edges)
            if len(tokens_graph.nodes) != len(nodes_graph.nodes):
                continue
            if nx.is_isomorphic(tokens_graph, nodes_graph):
                self.possible_token_pairs = {k: edges[i] for i, k in enumerate(self.possible_token_pairs)}
                self.nodes_tokens = {np[i]: {self.possible_token_pairs[np][i]} for np in self.possible_token_pairs for i
                                     in range(2)}
                return True
        return False

    def _match_constraints(self):
        """Checks if there is a token pair that matches all constraints"""

        for np in self.constraints:
            self.possible_token_pairs[np] = list(product(self.nodes_tokens[np[0]], self.nodes_tokens[np[1]]))
            for constraint in self.constraints[np]:
                if constraint == 'deprels':
                    self.possible_token_pairs[np] = self._pairs_matching_relpattern(np)
                elif constraint == 'lindist':
                    self.possible_token_pairs[np] = self._linear_distance(np)
                elif constraint == 'fconstraint':
                    self.possible_token_pairs[np] = self._feature_constraint(np)
                else:
                    raise ValueError('Wrong constraint type')
                if not self.possible_token_pairs[np]:
                    return False
                else:
                    self.nodes_tokens[np[0]] = set([p[0] for p in self.possible_token_pairs[np]])
                    self.nodes_tokens[np[1]] = set([p[1] for p in self.possible_token_pairs[np]])
        if not self._find_isomorphism():
            return False
        return True

    def filter_sentence(self, node_pattern: dict, constraints: dict):
        self.node_pattern = node_pattern
        self.constraints = constraints
        self.nodes_tokens = {node: [] for node in self.node_pattern}
        self.possible_token_pairs = {pair: set() for pair in self.constraints}
        if not self._find_all_nodes():
            return False
        else:
            self.sent_deprels = self.all_deprels(self.sentence)
            if self._match_constraints():
                return True
            else:
                return False
