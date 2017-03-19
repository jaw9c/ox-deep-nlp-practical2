import numpy as np
import os
from random import shuffle
import re
import urllib.request
import zipfile
import lxml.etree

def ted_talks_and_labels():

	# Download the dataset if it's not already there: this may take a minute as it is 75MB
	if not os.path.isfile('ted_en-20160408.zip'):
		urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")
	with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
		doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
	input_labels = doc.xpath('//keywords/text()')
	input_texts = doc.xpath('//content/text()')
	del doc
	def extract_ted_labels(labels):
		total = 0
		if 'technology' in labels:
			total += 1
		if 'education' in labels:
			total += 2
		if 'design' in labels:
			total += 4
		out = np.zeros(8)
		out[total] = 1
		return out

	def wordify(text):
		re.sub(r'\([^)]*\)', '', text)
		into_sentences = []
		for line in text.split('\n'):
			m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
			into_sentences.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
		sentences_wout_specials = [re.sub(r"[^a-z0-9]+", " ", sent.lower()).split() for sent in into_sentences ]
		return [word for sent in sentences_wout_specials for word in sent]

	talk_labels = [extract_ted_labels(labels.split(', ')) for labels in input_labels]
	talk_texts = [wordify(text) for text in input_texts]
	del input_labels
	del input_texts
	return talk_texts, talk_labels