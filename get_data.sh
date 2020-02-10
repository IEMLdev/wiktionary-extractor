#!/usr/bin/env bash

PYTHON=venv/bin/python
DATA_FOLDER=data
WIKTIONARY_DUMP=enwiktionary-latest-pages-articles

WIKTIONARY_DUMP_XML=$WIKTIONARY_DUMP.xml
WIKTIONARY_DUMP_XML_BZ2=$WIKTIONARY_DUMP_XML.bz2
WIKTIONARY_DUMP_JSON=$WIKTIONARY_DUMP.json

URL=https://dumps.wikimedia.org/enwiktionary/latest/$WIKTIONARY_DUMP_XML_BZ2

WIKTIONARY_DUMP_XML_FILE=$DATA_FOLDER/$WIKTIONARY_DUMP_XML
WIKTIONARY_DUMP_XML_BZ2_FILE=$DATA_FOLDER/$WIKTIONARY_DUMP_XML_BZ2
WIKTIONARY_DUMP_JSON_FILE=$DATA_FOLDER/$WIKTIONARY_DUMP_JSON

if [[ ! (-e $WIKTIONARY_DUMP_XML_BZ2_FILE) ]]; then
    echo "Downloading Wiktionary dump..."
    wget --directory-prefix $DATA_FOLDER $URL
fi
if [[ ! (-e $WIKTIONARY_DUMP_XML_FILE) ]]; then
    echo "Extracting Wiktionary dump..."
    bzip2 -dk $WIKTIONARY_DUMP_XML_BZ2_FILE
fi
if [[ ! (-e $WIKTIONARY_DUMP_JSON_FILE) ]]; then
    echo "Parsing Wiktionary dump..."
    $PYTHON wiktionaryExtractor.py $WIKTIONARY_DUMP_XML_FILE $WIKTIONARY_DUMP_JSON_FILE
fi

mongoimport --db wiktionary --collection $WIKTIONARY_DUMP --type json $WIKTIONARY_DUMP_JSON_FILE --json