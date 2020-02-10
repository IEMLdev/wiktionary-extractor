#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
# import os
import re, json
# from bs4 import BeautifulSoup
# import nltk
from venv import logger

from nltk.tokenize import word_tokenize
from tqdm import tqdm
from xml.etree import ElementTree as ET


def getDataLists():
	langs = set(["Abaza", "Abkhaz", "Adyghe", "Afrikaans", "Akan", "Albanian",
				 "American Sign Language", "Amharic", "Arabic", "Aragonese", "Armenian",
				 "Old Armenian", "Assamese", "Asturian", "Avar", "Aymara", "Azerbaijani",
				 "Bashkir", "Basque", "Belarusian", "Bengali", "Breton", "Bulgarian",
				 "Burmese", "Buryat", "Catalan", "Chakma", "Chechen", "Cherokee", "Chinese:",
				 "Cantonese", "Dungan", "Mandarin", "Chukchi", "Chuvash", "Cornish",
				 "Crimean Tatar", "Czech", "Danish", "Dargwa", "Dhivehi", "Dutch", "Eastern Mari",
				 "English", "Erzya", "Esperanto", "Estonian", "Evenki", "Faroese",
				 "Finnish", "Franco-Provençal", "French", "Middle French", "Fula", "Gagauz",
				 "Galician", "Georgian", "German", "Greek", "Ancient", "Greenlandic",
				 "Guerrero Amuzgo", "Gujarati", "Haitian Creole", "Hausa", "Hawaiian",
				 "Hebrew", "Hiligaynon", "Hindi", "Hopi", "Hungarian", "Icelandic", "Ido",
				 "Igbo", "Indonesian", "Ingush", "Interlingua", "Interlingue", "Inuktitut",
				 "Irish", "Italian", "Japanese", "Javanese", "Kabardian", "Kalmyk", "Kannada",
				 "Kapampangan", "Karachay-Balkar", "Karakalpak", "Karelian", "Kashubian",
				 "Kazakh", "Khakas", "Khmer", "Kikuyu", "Komi-Permyak", "Korean", "Kumyk",
				 "Kurdish:", "Kurmanji", "Sorani", "Kven", "Kyrgyz", "Lak", "Lao", "Latgalian",
				 "Latin", "Latvian", "Lezgi", "Ligurian", "Limburgish", "Lithuanian",
				 "Low German", "Luhya", "Lule Sami", "Luxembourgish",
				 "Macedonian", "Malagasy", "Malay", "Malayalam", "Maltese", "Manchu", "Manx",
				 "Maori", "Marathi", "Mauritian Creole", "Meru", "Middle Persian", "Moksha",
				 "Mongolian:", "Cyrillic", "Mongolian", "Nahuatl", "Navajo", "Neapolitan",
				 "Nenets:", "Tundra Nenets", "Nepali", "Nogai", "Norman", "Northern Sami",
				 "Norwegian:", "Bokmål", "Nynorsk", "Novial", "Occitan", "Ojibwe", "Oriya",
				 "Ossetian:", "Digor", "Iron", "Papiamentu", "Pashto", "Persian", "Plautdietsch",
				 "Polish", "Portuguese", "Punjabi", "Quechua", "Romanian", "Romansch", "Russian",
				 "Rusyn", "Rwanda-Rundi", "Samogitian", "Sanskrit", "Sardinian", "Scots",
				 "Scottish Gaelic", "Serbo-Croatian:", "Cyrillic", "Roman", "Shona", "Shor",
				 "Sicilian", "Silesian", "Sindhi", "Sinhalese", "Skolt Sami", "Slovak",
				 "Slovene", "Somali", "Sorbian:", "Lower Sorbian", "Upper Sorbian", "Sotho",
				 "Southern Altai", "Southern Ohlone", "Southern Sami", "Spanish", "Swahili",
				 "Swedish", "Tabasaran", "Tagalog", "Tajik", "Tamil", "Tatar", "Telugu",
				 "Thai", "Tibetan", "Tigrinya", "Tok Pisin", "Turkish", "Turkmen", "Tuvan",
				 "Udmurt", "Ukrainian", "Urdu", "Uyghur", "Uzbek", "Cyrillic", "Venda",
				 "Veps", "Vietnamese", "Volapük", "Võro", "Wallisian", "Walloon", "Welsh",
				 "West Frisian", "Western Cham", "Wolof", "Xhosa", "Yakut", "Yiddish",
				 "Yoruba", "Zazaki", "Zhuang", "Zulu"])

	sections = ["Anagrams", "Alternative forms", "Antonyms", "Declension", "Descendants",
				"Derived terms", "Economie", "Etymology", "Etymology ([\d]+)", "Further reading",
				"Hyponyms", "Inflection", "Mutation", "Pronunciation", "Quotations",
				"References", "Related terms", "Romanization", "Synonyms", "Translations",
				"Usage notes"]
	tokenNature = ["Adverb", "Adjective", "Interjection", "Noun", "Preposition", "Verb"]
	return langs, sections, tokenNature


def cleanDescription(description):
	# get hyperlinks
	hyperlinks = [h.split("|")[0] for h in set(re.findall(r"\[\[(.+?)\]\]", description))]
	cleanDescription = " ".join([d.split("|")[-1] for d in re.split(r"[\[\{]{2}(.+?)[\]\}]{2}", description)]).replace(
		"\'", "")
	cleanDescription = re.sub(r"<!--.+?-->", "", cleanDescription)
	descriptionSplit = re.split(r"\n# ", cleanDescription)
	synsetData = []
	for descrp in descriptionSplit:
		synSplit = re.split(r"\n#", descrp)
		synDescrip = synSplit[0] if synSplit[0][:2] != "# " else synSplit[0][2:]
		synSynonyms = [s for s in synSplit[1:] if s[:3] == ":  "]
		synSynonyms = synSynonyms[0][3:] if synSynonyms != [] else []
		synSynonyms = synSynonyms if synSynonyms[-2:] != "  " else synSynonyms[:-2]
		synSynonyms = synSynonyms if synSynonyms[-1:] != " " else synSynonyms[:-1]
		synSynonyms = synSynonyms if synSynonyms[:2] != "  " else synSynonyms[2:]
		synSynonyms = synSynonyms if synSynonyms[:1] != " " else synSynonyms[1:]
		# synCitation = [s for s in synSplit[1:] if s[:3] == "*  "][0]
		if synDescrip != "":
			synsetData.append([synDescrip, synSynonyms])
	return synsetData, hyperlinks


def cleanTxt(txt, keepMetaData=False):
	if type(txt) is list:
		if len(txt) == 1:
			txt = txt[0]
		else:
			return [cleanTxt(t, keepMetaData) for t in txt]
	if keepMetaData is False:
		cleanedTxt = " ".join([d.split("|")[-1] for d in re.split(r"[\[\{]{2}(.+?)[\]\}]{2}", txt)]).replace("\'", "")
	else:
		cleanedTxt = " ".join([d.split("|")[-1] for d in re.split(r"[\[]{2}(.+?)[\]]{2}", txt)]).replace("\'", "")
		cleanedTxt = re.sub(r"\{\{|\}\}", "", cleanedTxt)
		cleanedTxt = cleanedTxt.replace("|", " & ")
	cleanedTxt = re.sub(r"<.+?>", "", cleanedTxt)
	cleanedTxt = re.sub(r"\n\n|\n", " ", cleanedTxt)
	cleanedTxt = cleanedTxt if cleanedTxt[-2:] != "  " else cleanedTxt[:-2]
	cleanedTxt = cleanedTxt if cleanedTxt[-1:] != " " else cleanedTxt[:-1]
	cleanedTxt = cleanedTxt if cleanedTxt[:2] != "  " else cleanedTxt[2:]
	cleanedTxt = cleanedTxt if cleanedTxt[:1] != " " else cleanedTxt[1:]
	if re.match(r"\*  |\* ", cleanedTxt) is not None:
		cleanedTxt = [s.replace("*  ", "").replace("* ", "") for s in re.split(r"\n\*[ ]{1,2}", cleanedTxt)]
		cleanedTxt = [s[0] if len(s) == 1 else s for s in cleanedTxt]
	return cleanedTxt


def findAll(inputFile, tag):
	last_elem = None
	for event, elem in ET.iterparse(inputFile):
		if elem.tag == tag:
			if last_elem:
				yield last_elem
				# the xml tree walk go to the parent before the children, so the children aren't loaded yet for elem.
				# we wait for all the children to be loaded first, and return the last_elem (we bet on the fact that the
				# elements we are looking for are not parent of each others)
				last_elem.clear()
			last_elem = elem

	if last_elem:
		yield last_elem


def extractAndDump(inputPath, outputPath="./output.json"):
	dataDict = {}
	# get uniform wiktionary titles and data
	langs, sections, tokenNature = getDataLists()
	# open xml file
	ns = '{http://www.mediawiki.org/xml/export-0.10/}'

	# open output path - one json per line

	# with open(inputPath) as xmlFile:
	# 	xmlContent = xmlFile.read()
	# 	soup = BeautifulSoup(xmlContent, "xml")
	# 	# divide into the pages contents
	# 	wiktioPages = soup.find_all("page")
	total_page_count = sum(1 for _ in findAll(inputPath, ns + 'page'))

	for page in tqdm(findAll(inputPath, ns + 'page'), total=total_page_count):
		# if not page.find_all(".//" + ns + "title") or not page.find_all(".//" + ns + "text"):
		# 	continue
		# else:
		# 	print("not miss title or text")

		wordEntry = re.sub(r"<title>|</title>", "", page.findall(".//" + ns + "title")[0].text)
		pageText = page.findall(".//" + ns + "text")[0].text
		if not pageText:
			logger.error("No page text in element page: " + ET.tostring(page, encoding='unicode', method='xml'))
			continue

		# get lang specific content
		langSplitText = re.split("({0})".format("|".join(["=={0}==".format(l) for l in langs])), pageText)
		# separate by the languages of interest
		for sectionInd, txtSection in enumerate(list(langSplitText)):
			# get the languages of interest
			if txtSection in ["==English==", "==French=="]:
				sectionLang = txtSection.replace("==", "")
				if sectionLang not in dataDict:
					dataDict[sectionLang] = {}
				# get the information
				sectionContent = langSplitText[sectionInd + 1]
				# prepare the variables that will contain the language/synset data
				sectEtymology = []
				sectMorphSyntx = {}
				sectPronunciation = []
				sectSynonyms = []
				sectAntomyms = []
				sectHyponyms = []
				sectDerivedTerms = []
				sectRelatedTerms = []
				sectTranslations = []
				# divide the section content into subsections
				subsectionSplit = re.split(r"(======[\n]?|=====[\n]?|====[\n]?|===[\n]?|==[\n]?)", sectionContent)
				for subsectionInd, subsection in enumerate(list(subsectionSplit[:-2])):
					if "==" in subsection and "==\n" in subsectionSplit[subsectionInd + 2]:
						# get the title
						subsectionTitle = subsectionSplit[subsectionInd + 1]
						subsectionContent = subsectionSplit[subsectionInd + 3]
						# capture etymology
						if "Etymology" in subsectionTitle:
							subsectionContent = cleanTxt(subsectionContent, keepMetaData=True)
							if subsectionContent != "":
								sectEtymology.append(subsectionContent)
						# capture morphosyntactic class (POS, grammatical nature of the word)
						elif subsectionTitle in tokenNature:
							if subsectionTitle not in sectMorphSyntx:
								sectMorphSyntx[subsectionTitle] = []
							sectMorphSyntx[subsectionTitle].append(subsectionContent)
						# capture pronunciation
						elif subsectionTitle in ["Pronunciation"]:
							sectPronunciation.append(
								[s for s in subsectionContent.replace("\n", "").split("* ") if s != ""])
						# capture synonyms
						elif subsectionTitle in ["Alternative forms", "Synonyms"]:
							sectSynonyms.append(cleanTxt(subsectionContent))
						# capture antonyms
						elif subsectionTitle in ["Antonyms"]:
							sectAntomyms.append(cleanTxt(subsectionContent))
						# capture hyponyms
						elif subsectionTitle in ["Hyponyms"]:
							subsectionContent = re.split(r"[ ]?[a-z]{2}[ ]{1,2}\*[ ]{1,2}-", subsectionContent)
							sectHyponyms.append(cleanTxt(subsectionContent))
						# capture derived terms
						elif subsectionTitle in ["Derived terms"]:
							sectDerivedTerms.append(cleanTxt(subsectionContent))
						# capture related terms
						elif subsectionTitle in ["Related terms"]:
							sectRelatedTerms.append(cleanTxt(subsectionContent))
						# capture translations
						elif subsectionTitle in ["Translations"]:
							cleanedTxt = cleanTxt(subsectionContent)
							if isinstance(cleanedTxt, list) and len(cleanedTxt) == 1:
								cleanedTxt = cleanedTxt[0]

							sectTranslations.append(cleanedTxt.split(" * "))
					# print(subsection)
					# print(repr(subsectionTitle))
				# get the POS, synonyms and particular
				for synsetPos, synsetContent in sectMorphSyntx.items():
					for content in synsetContent:
						for subcontent in content.split("\n\n"):
							if subcontent != "" and re.fullmatch(r"\{\{.+\}\}", subcontent) is None:
								synsetData, hyperlinks = cleanDescription(subcontent)
								# save into dict
								for indexSynset, (definitionSynset, synonymSynset) in enumerate(synsetData):
									dataDict[sectionLang]["{0}{1}".format(wordEntry, indexSynset)] = {
										"language": sectionLang,
										"word": wordEntry,
										"definition": definitionSynset,
										"definition token": word_tokenize(definitionSynset, language=sectionLang.lower()),
										"inter-reference in definition": hyperlinks,
										"synset synonyms": synonymSynset,
										"word synonyms": sectSynonyms,
										"etymology": sectEtymology,
										"pronunciation": sectPronunciation,
										"antonyms": sectAntomyms,
										"hyponyms": sectHyponyms,
										"derived terms": sectDerivedTerms,
										"translations": sectTranslations,
										"related terms": sectRelatedTerms}


	# dump
	with open(outputPath, u'w', encoding=u'utf8') as dictFile:
		dictFile.write('')
		json.dump(dataDict, dictFile)
	return dataDict

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("inputPath", type=str, help="path to a wiktionary dump")
	parser.add_argument("outputPath", type=str, help="path to the outputed json file")

	args = parser.parse_args()

	inputPath = args.inputPath
	outputPath = args.outputPath
	extractAndDump(inputPath, outputPath)
