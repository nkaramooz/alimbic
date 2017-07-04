import xml.etree.ElementTree as ET
import sys

tree = ET.parse('test.xml')

root = tree.getroot()

ns = {'real_person': 'http://people.example.com',
      'role': 'http://characters.example.com'}




#'real_person:actors', ns
for actor in root.iter('{http://people.example.com}actor'):
    name = actor.find('real_person:name', ns)
    print name.text
    for char in actor.findall('role:character', ns):
        print ' |-->', char.text