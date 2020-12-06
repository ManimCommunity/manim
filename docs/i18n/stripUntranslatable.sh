#!/bin/bash
rm translatable.po
rm untranslatable.po
for i in `find en/ -name "*.po"`
do
	awk -f stripUntranslatable.awk -i inplace $i
	cat $i >> translatable.po
	echo "OK: $i"
done
