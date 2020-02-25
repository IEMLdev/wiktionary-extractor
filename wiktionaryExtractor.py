#!/usr/bin/python
# -*- coding:utf-8 -*-  

import argparse
import re, json
from venv import logger
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from xml.etree import ElementTree as ET

import nltk
nltk.download("punkt")

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
	tokenPos= ["Adverb", "Adjective", "Interjection", "Noun", "Preposition", "Verb"]
	return langs, sections, tokenPos


def cleanDescription(description):
	# get hyperlinks
	hyperlinks = [h.split("|")[0] for h in set(re.findall(r"\[\[(.+?)\]\]", description))]
	cleanDescription = " ".join([d.split("|")[-1] for d in re.split(r"[\[\{]{2}(.+?)[\]\}]{2}", description)]).replace("\'", "")
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


def cleanText(txt, keepMetaData=False):
	if type(txt) is list:
		if len(txt) == 1:
			txt = txt[0]
		else:
			return [cleanText(t, keepMetaData) for t in txt]
	cleanTxt = txt
	if keepMetaData is False:
		cleanTxt = " ".join([d.split("|")[-1] for d in re.split(r"[\[\{]{2}(.+?)[\]\}]{2}", txt)]).replace("\'", "")
	# remove double brackets (open or close) if the contrary one is not present
	for brkts in [["[[", "]]"], ["{{", "}}"], ["((", "))"]]:
		if brkts[0] in cleanTxt and brkts[1] not in cleanTxt:
			cleanTxt = cleanTxt.replace(brkts[0], "")
		elif brkts[1] in cleanTxt and brkts[0] not in cleanTxt:
			cleanTxt = cleanTxt.replace(brkts[1], "")
	cleanTxt = re.sub(r"<.+?>", "", cleanTxt)
	cleanTxt = re.sub(r"\n\n|\n", " ", cleanTxt)
	cleanTxt = cleanTxt if cleanTxt[-2:] != "  " else cleanTxt[:-2]
	cleanTxt = cleanTxt if cleanTxt[-1:] != " " else cleanTxt[:-1]
	cleanTxt = cleanTxt if cleanTxt[:2] != "  " else cleanTxt[2:]
	cleanTxt = cleanTxt if cleanTxt[:1] != " " else cleanTxt[1:]
	if re.match(r"\*  |\* ", cleanTxt) is not None:
		cleanTxt = [s.replace("*  ", "").replace("* ", "") for s in re.split(r"\n\*[:]?[ ]{1,2}", cleanTxt)]
		cleanTxt = [s[0] if len(s) == 1 else s for s in cleanTxt]
	return cleanTxt


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
	langs, sections, tokenPos = getDataLists()
	# open xml file
	ns = '{http://www.mediawiki.org/xml/export-0.10/}'
	total_page_count = sum(1 for _ in findAll(inputPath, ns + 'page'))
	for page in tqdm(findAll(inputPath, ns + 'page'), total=total_page_count):
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
				sectionContent = langSplitText[sectionInd+1]
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
					if "==" in subsection and "==\n" in subsectionSplit[subsectionInd+2]:
						# get the title
						subsectionTitle = subsectionSplit[subsectionInd+1]
						subsectionContent = subsectionSplit[subsectionInd+3]
						# capture etymology
						if "Etymology" in subsectionTitle:
							subsectionContent = cleanText(subsectionContent, keepMetaData=True)
							if subsectionContent != "":
								sectEtymology.append(subsectionContent)
						# capture morphosyntactic class (POS, grammatical nature of the word)
						elif subsectionTitle in tokenPos:
							if subsectionTitle not in sectMorphSyntx:
								sectMorphSyntx[subsectionTitle] = []
							sectMorphSyntx[subsectionTitle].append(subsectionContent)
						# capture pronunciation
						elif subsectionTitle in ["Pronunciation"]:
							sectPronunciation.append([s for s in subsectionContent.replace("\n", "").split("* ") if s != ""])
						# capture synonyms
						elif subsectionTitle in ["Alternative forms", "Synonyms"]:
							sectSynonyms.append(cleanText(subsectionContent))
						# capture antonyms
						elif subsectionTitle in ["Antonyms"]:
							sectAntomyms.append(cleanText(subsectionContent))
						# capture hyponyms
						elif subsectionTitle in ["Hyponyms"]:
							subsectionContent = re.split(r"[ ]?[a-z]{2}[ ]{1,2}\*[ ]{1,2}-", subsectionContent)
							sectHyponyms.append(cleanText(subsectionContent))
						# capture derived terms
						elif subsectionTitle in ["Derived terms"]:
							sectDerivedTerms.append(cleanText(subsectionContent))
						# capture related terms
						elif subsectionTitle in ["Related terms"]:
							sectRelatedTerms.append(cleanText(subsectionContent))
						# capture translations
						elif subsectionTitle in ["Translations"]:
							if type(subsectionContent) is list:
								if len(subsectionContent) == 1:
									subsectionContent = subsectionContent[0]
							subsectionContentList = [s for s in re.split(r"\{\{trans-top\|?|\{\{checktrans-top\|?|\{\{trans-bottom\}\}|[\n]{2,6}", subsectionContent) if s not in [""]]
							for subCont in subsectionContentList:
								subCont = re.sub(r"{{trans-mid}}[\n]?", "", subCont)
								translDef = [el for el in subCont.split("\n") if ("}}" in el and "{{" not in el)]
								translDef = "" if len(translDef) == 0 else translDef[0].replace("}}", "")
								# save to dict
								transDict = {"definition": translDef}
								translContents = re.split(r"\n(?=\*)", subCont)
								translContents = translContents[1:]
								for trContent in translContents:
									trLang = re.match(r"\*[ :]{1,2}.+?:[ ]?", trContent)
									if trLang not in [None, ""]:
										trContent = trContent.replace(trLang.group(0), "")
										trLang = re.sub(r"\*[ :]{1,2}|: ", "", trLang.group(0))
										transDict[trLang] = cleanText(trContent, keepMetaData=True)
								sectTranslations.append(transDict)
				# get the POS, links, synonyms and other particular data
				for synsetPos, synsetContent in sectMorphSyntx.items():
					wikiPos = []
					for content in synsetContent:
						for subcontent in content.split("\n\n"):
							if subcontent != "":
								# get synset data
								if re.fullmatch(r"\{\{.+\}\}", subcontent) is None:
									synsetData, hyperlinks = cleanDescription(subcontent)
									# save into dict
									for indexSynset, (definitionSynset, synonymSynset) in enumerate(synsetData):
										dataDict[sectionLang]["{0}_{1}_{2}".format(wordEntry, synsetPos, indexSynset)] = {
											"language": sectionLang,
											"word": wordEntry,
											"pos data": wikiPos,
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
											"related terms":sectRelatedTerms}
								# get metadata
								else:
									wikiPos = subcontent.replace("{{en-", "").replace("{{fr-", "").replace("{{", "").replace("}}", "").split("|")
									

	# dump
	with open(outputPath, u'w', encoding=u'utf8') as dictFile:
		dictFile.write('')
		json.dump(dataDict, dictFile)
	return dataDict


# inputPath = "./wiktionary-dumps/enwiktionary-20191120-pages-articles-multistream.EXTRACT.xml"
# # inputPath = "./wiktionary-dumps/enwiktionary-20191120-pages-meta-current.EXTRACT.xml"
# outputPath = "./wiktionary-dumps/enwiktionary-20191120.json"
# extractAndDump(inputPath, outputPath)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-in", "--inputPath", type=str, help="path to a wiktionary dump")
	parser.add_argument("-out", "--outputPath", type=str, help="path to the outputed json file")
	args = parser.parse_args()

	inputPath = args.inputPath
	outputPath = args.outputPath
	extractAndDump(inputPath, outputPath)
