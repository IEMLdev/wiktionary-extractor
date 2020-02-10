import argparse

from pymongo import MongoClient, IndexModel
import os
from tqdm import tqdm
import json

def load_json(inputFile):
	with open(inputFile) as fp:
		return json.load(fp)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("inputPath", type=str, help="path to a wiktionary dump")
	# parser.add_argument("outputPath", type=str, help="path to the outputed json file")

	args = parser.parse_args()

	inputPath = args.inputPath
	# outputPath = args.outputPath
	# extractAndDump(inputPath, outputPath)

	collection = MongoClient()['wiktionary'][os.path.basename(inputPath).split('.')[0]]
	collection.drop()
	collection.drop_indexes()

	collection.create_indexes([
		IndexModel([('language', 1), ('synset', 1)])
	])
	data = load_json(inputPath)

	for i, (lang, v) in enumerate(data.items()):
		for synset, doc in tqdm(v.items(), "Importing {} ({}/{})".format(lang, i, len(data))):
			collection.insert_one({
				'language': lang,
				'synset': synset,
				**doc
			})
