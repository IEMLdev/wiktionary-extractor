#!/usr/bin/python
# -*- coding:utf-8 -*-

import json
import ieml
import numpy
import argparse
from itertools import chain
from ieml.usl.usl import usl
from ieml.usl.word import Word
from bert_embedding import BertEmbedding

class WiktionaryData:
    """
    Modify the wiktionary extracted data (as json) to output a vector representation
    """

    def __init__(self, in_file_path, out_file_path):
        # input file
        self.json_in_path = in_file_path
        self.json_file = open(in_file_path)
        # output file
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
        # bert model
        self.bert = BertEmbedding(model="bert_12_768_12")

    def reset_json_in_file(self):
        self.json_file.close()
        self.json_file = open(self.json_in_path)

    def close_all(self):
        self.json_file.close()
        self.out_file.close()
        self.vocab_file.close()

    def make_bert_emb_list_file(self):
        # overwrtite data from previous files
        sent_file_path = self.json_in_path.replace(".json", ".sent")
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
                sent_file.write("{0}\n".format(wikt_sent))
                # next
                ln = self.json_file.readline()
                counter += 1 ###################
                if counter == 500:
                    break
        self.reset_json_in_file()
        # open the sentence file and transform it to bert embeddings
        with open(sent_file_path) as sent_file:
            sent_emb = self.bert([s.replace("\n", "") for s in sent_file.readlines()])
        # # dump the embeddings
        # for (bert_vocab, bert_vect) in sent_emb:
        #     self.vocab_file.write("{0}\n".format(json.dumps(bert_vocab)))
        #     try:
        #         self.out_file.write("{0}\n".format(json.dumps(bert_vect)))
        #         print(11111, type(bert_vect), bert_vect[0])
        #     except TypeError:
        #         self.out_file.write("{0}\n".format(bert_vect.dumps()))
        #         print(22222, type(bert_vect), bert_vect[0])
        # # numpy.save(self.out_file, sent_emb)
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
    from ieml.ieml_database import IEMLDatabase
    def __init__(self, database_folder_path):
        self.database = IEMLDatabase(folder=database_folder_path)
        # bert model
        self.bert = BertEmbedding(model="bert_12_768_12")

    def get_word_objects(self):
        return self.database.list(parse=False, type='word')

    def list_polymorpheme_of_word(w):
        w = usl(w)
        assert isinstance(w, Word)
        return list(chain.from_iterable((sfun.actor.pm_content, sfun.actor.pm_flexion)
                                        for sfun in w.syntagmatic_fun.actors.values()))

    def get_natural_lang_meanings(self, lang="en"):
        descriptors = self.database.get_descriptors()
        for word in self.get_word_objects():
            polymorphemes = list_polymorpheme_of_word(word)
            nl_meanings = []
            for poly in polymorphemes:
                try:
                    nl_meanings.append(descriptors.get_values(poly, language=lang, descriptor='translations'))
                except ValueError:
                    pass
            yield nl_meanings
            
    def get_bert_emb(self, string):
        return self.bert([string])

def choose_task(t_task=None, input_path=None, output_path=None):
    """
    Choose and launch a task.
    :param t_task:
    :param input_path:
    :param output_path:
    :return:
    """
    wiki_data = WiktionaryData(input_path, output_path)
    ieml_data = IemlData("./ieml-language-master/")
    if t_task.lower() in ["json2list", "list", "json", "makelist"]:
        wiki_data.make_bert_emb_list_file()
        #######################
        for string in ieml_data.get_natural_lang_meanings(lang="en"):
            repri = ieml_data.get_bert_emb(string)
            print(repri)
        #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, default="None", help="path to the wiktionary json")
    parser.add_argument("-in", "--input", type=str, help="path to the wiktionary json")
    parser.add_argument("-out", "--output", type=str, help="path to the output file")
    args = parser.parse_args()

    a_task = args.task
    input_path = args.input
    output_path = args.output

    choose_task(a_task, input_path, output_path)

    # COMMAND
    # python3 natural_lang2vector.py -t makelist -in ./wiktionary-dumps/wiktionary_English.json -out embeddings/wiktionary_English_embedd.tsv