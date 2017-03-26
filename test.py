# -*- coding: utf-8 -*-
__author__ = 'pangbochen'

from xml.dom.minidom import parse
import xml.dom.minidom
DOMTree = xml.dom.minidom.parse("1500004.xml")
collection = DOMTree.documentElement
movies = collection.getElementsByTagName("AbstractNarration")
movie = movies[0]
print(movie.childNodes[0].data)
print('end')