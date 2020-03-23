#!/usr/bin/python
# -*- coding:utf-8 -*-

import json
import ieml
import numpy
import argparse
import numpy as np
from itertools import chain
from ieml.usl.usl import usl
from ieml.usl.word import Word
from bert_embedding import BertEmbedding
from sklearn.neighbors import NearestNeighbors

class BertEmbedd:

    def __init__(self):
        # bert model
        self.bert = BertEmbedding(model="bert_12_768_12")

    def removeNanInfAndOvermax(self, an_array):
        max_supported = np.finfo(np.float64).max
        # remove NaN
        if np.isnan(an_array.any()):
            an_array = np.nan_to_num(an_array)
        # remove inf & -inf
        if np.isfinite(an_array.all()) is False:
            an_array = np.array([f if f != float("inf") else max_supported for f in an_array])
            an_array = np.array([f if f != float("-inf") else (-1 * max_supported) for f in an_array])
        # remove overmax
        an_array = np.array([f if f < max_supported else max_supported for f in an_array])
        an_array = np.array([f if f > (-1 * max_supported) else (-1 * max_supported) for f in an_array])
        return an_array

    def fuse(self, word_emb):
        new = np.average(word_emb[1], axis=0)
        # reduce the size of float to the maximum supported by float64
        new = self.removeNanInfAndOvermax(np.array(new))
        return new

    def find_nearest(self, a_array, array_of_arrays, n=1):
        n = len(array_of_arrays) if n > len(array_of_arrays) else n
        a_array = a_array.reshape(1, -1)
        # get 1 nearest neighbor using "ball tree" ('auto', 'ball_tree', 'kd_tree', 'brute')
        potential_neighbors = NearestNeighbors(n_neighbors=n, algorithm='ball_tree')
        potential_neighbors = potential_neighbors.fit(array_of_arrays)
        # potential_neighbors = potential_neighbors.fit_transform(array_of_arrays)
        distances, indices = potential_neighbors.kneighbors(a_array)
        indices = indices[0].tolist()
        for i, (dist, ind) in enumerate(zip(distances[0], indices)):
            indices[i] = ind if dist != float("inf") else None
        return indices

class WiktionaryData:
    """
    Modify the wiktionary extracted data (as json) to output a vector representation
    """

    def __init__(self, in_file_path, out_file_path=None, in_ieml_file_path=None):
        # input file
        self.json_in_path = in_file_path if in_file_path is not None else in_ieml_file_path
        self.json_file = open(self.json_in_path)
        # output file
        if out_file_path is not None:
            with open(out_file_path, "w") as output_file:
                output_file.write("")
            self.out_file_path = out_file_path
            self.out_file = open(out_file_path, "a")
            # vocabulary file
            vocab_path = "{0}.vocab".format(out_file_path.replace(".tsv", ""))
            with open(vocab_path, "w") as output_vocab_file:
                output_vocab_file.write("")
            self.vocab_file_path = vocab_path
            self.vocab_file = open(vocab_path, "a")

    def reset_json_in_file(self):
        self.json_file.close()
        self.json_file = open(self.json_in_path)

    def close_all(self):
        self.json_file.close()
        self.out_file.close()
        self.vocab_file.close()

    def make_bert_emb_list(self, bert_class=None, dump=False):
        if bert_class is None:
            bert_class = BertEmbedd()
        sent_file_path = self.json_in_path.replace(".json", ".sent")
        # overwrtite data from previous files if dump is required
        if dump is not False:
            with open(sent_file_path, "w") as sent_file:
                sent_file.write("")
        with open(sent_file_path, "a") as sent_file:
            # get the data
            ln = self.json_file.readline()
            counter = 0 ##################
            while ln:
                data_dict = json.loads(ln.replace("\n", ""))
                wikt_sent = data_dict["word"]
                for section in ["antonyms", "definition", "synset synonyms", "word synonyms", "hyponyms",
                                "derived terms", "related terms"]:
                    if section in data_dict and len(data_dict[section]) != 0:
                        if section == "antonyms" and len(data_dict[section]) != 0:
                            antonym_data = [a if type(a) is str else "\t".join(a) for a in data_dict[section]]
                            antonym_data = [a for a in ("\t".join(antonym_data)).split("\t") if a != ""]
                            wikt_sent += " not ".join(antonym_data)
                        elif type(data_dict[section]) is list:
                            list_data = [sl if type(sl) is str else " ".join(sl) for sl in data_dict[section]]
                            wikt_sent += " ".join(list_data).replace("\n", " ")
                        elif type(data_dict[section]) is str:
                            wikt_sent += " {0}".format(data_dict[section]).replace("\n", "")
                # dump wiktionary "sent" (useful definition data)
                if dump is not False:
                    sent_file.write("{0}\n".format(wikt_sent))
                # next
                ln = self.json_file.readline()
                counter += 1 ###################
                # if counter == 50:
                #     break
        self.reset_json_in_file()
        # open the sentence file and transform it to bert embeddings
        with open(sent_file_path) as sent_file:
            sent_emb = bert_class.bert([s.replace("\n", "") for s in sent_file.readlines()])
        if dump is not False:
            # dump the embeddings
            for (bert_vocab, bert_vect) in sent_emb:
                bert_vect = np.array(bert_vect)
                self.vocab_file.write("{0}\n".format(json.dumps(bert_vocab)))
                try:
                    self.out_file.write("{0}\n".format(json.dumps(bert_vect)))
                except TypeError:
                    self.out_file.write("{0}\n".format(bert_vect.dumps()))
            # numpy.save(self.out_file, sent_emb)
        self.close_all()
        return sent_emb

    def load_embeddings(self):
        emb_data = numpy.loadtxt(self.json_in_path, delimiter="\t")
        self.close_all()
        return emb_data


class IemlData:
    """
    Modify the ieml data to output a vector representation
    """

    def __init__(self, input_database_folder_path, out_file_path=None):
        from ieml.ieml_database import IEMLDatabase
        # input file
        self.database = IEMLDatabase(folder=input_database_folder_path)
        # output file
        if out_file_path is not None:
            with open(out_file_path, "w") as output_file:
                output_file.write("")
            self.out_file_path = out_file_path
            self.out_file = open(out_file_path, "a")

    def close_all(self):
        self.out_file.close()

    def get_word_objects(self):
        return self.database.list(parse=False, type='word')

    def list_polymorpheme_of_word(self, w):
        w = usl(w)
        assert isinstance(w, Word)
        return list(chain.from_iterable((sfun.actor.pm_content, sfun.actor.pm_flexion)
                                        for sfun in w.syntagmatic_fun.actors.values()))

    def get_natural_lang_meanings(self, lang="en"):
        word_nl_meanings = []
        descriptors = self.database.get_descriptors()
        for word in self.get_word_objects():
            polymorphemes = self.list_polymorpheme_of_word(word)
            nl_meanings = []
            for poly in polymorphemes:
                try:
                    nl_meanings.append(descriptors.get_values(poly, language=lang, descriptor='translations'))
                except ValueError:
                    pass
            # yield nl_meanings
            word_nl_meanings.append(nl_meanings)
        return word_nl_meanings
            
    def get_bert_emb(self, string, bert_class):
        bert_class = bert_class if bert_class is not None else BertEmbedd()
        return bert_class.bert([string])

    def make_bert_emb_list(self, lang="en", bert_class=None, dump=False):
        bert_class = bert_class if bert_class is not None else BertEmbedd()
        bert_embeddings = []
        for ieml_pm_in_nl in self.get_natural_lang_meanings(lang):
            ieml_pm_in_nl = " ".join([" ".join(pm) for pm in ieml_pm_in_nl if len(pm) != 0])
            # yield self.get_bert_emb(ieml_pm_in_nl, bert_class)
            bert_embeddings.append(self.get_bert_emb(ieml_pm_in_nl, bert_class))
        return bert_embeddings


def choose_task(t_task=None, input_path=None, output_path=None, input_ieml_path=None):
    """
    Choose and launch a task.
    :param t_task:
    :param input_path:
    :param output_path:
    :return:
    """
    bert_class = BertEmbedd()
    # COMMAND:
    # python3 natural_lang2vector.py -t findnearest -in ./wiktionary-dumps/wiktionary_English.json -indb ./ieml-language-master/
    if t_task.lower() in ["find", "findnear", "findnearest", "nearest"]:
        wiki_data = WiktionaryData(input_path, output_path)
        ieml_data = IemlData(input_ieml_path)
        # make the embeddings from scratch
        wiktionary_vector_data = wiki_data.make_bert_emb_list(bert_class)
        # load the embeddings from file
        # TODO FUNCT LOAD
        ##############################
        wiktionary_sents = [se[0] for se in wiktionary_vector_data if len(se[0]) != 0]
        wiktionary_embeddings = np.array([bert_class.fuse(se) for se in wiktionary_vector_data if len(se[0][0]) != 0])
        ieml_vector_data = ieml_data.make_bert_emb_list(lang="en", bert_class=bert_class)
        ieml_sents = [se[0][0] for se in ieml_vector_data if len(se[0][0]) != 0]
        ieml_embeddings = np.array([bert_class.fuse(se[0]) for se in ieml_vector_data if len(se[0][0]) != 0])
        for i, w in enumerate(ieml_embeddings):
            close_neighb_ind = bert_class.find_nearest(w, wiktionary_embeddings, n=1)
            print(111111, i, ieml_sents[i])
            print(2222222, close_neighb_ind[0], wiktionary_sents[close_neighb_ind[0]])
        ############################################
    # COMMAND:
    # python3 natural_lang2vector.py -t dumpwiktio -in ./wiktionary-dumps/wiktionary_English.json -out embeddings/wiktionary_English_embedd.tsv
    elif t_task.lower() in ["json2list", "dumpwiktio", "json", "dumpw"]:
        wiki_data = WiktionaryData(input_path, output_path, input_ieml_path)
        wiktionary_vector_data = wiki_data.make_bert_emb_list(bert_class, dump=True)
    # COMMAND:
    # python3 natural_lang2vector.py -t dumpwiktio -in ./ieml-language-master/ -out embeddings/ieml_English_embedd.tsv
    elif t_task.lower() in ["ieml2list", "dumpieml", "ieml", "dumpi"]:
        ieml_data = IemlData(input_path, output_path, input_ieml_path)
        ieml_vector_data = ieml_data.make_bert_emb_list(lang="en", bert_class=bert_class, dump=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, default="None", help="path to the wiktionary json")
    parser.add_argument("-in", "--input", type=str, default="None", help="path to the wiktionary json")
    parser.add_argument("-indb", "--inputDatabase", type=str, default="None", help="path to the ieml database folder")
    parser.add_argument("-out", "--output", type=str, help="path to the output file")
    args = parser.parse_args()

    a_task = args.task
    input_path = args.input
    input_path = None if input_path in ["None", "none"]
    input_ieml_path = args.inputDatabase
    input_ieml_path = None if input_path in ["None", "none"]
    output_path = args.output
    output_path = None if input_path in ["None", "none"]

    choose_task(a_task, input_path, output_path, input_ieml_path)
