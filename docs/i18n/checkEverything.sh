#!/bin/bash
rm untranslatable.po
for i in `find en/ -name "*.po"`
do
	awk -f stripUntranslatable.awk -i inplace $i
	echo "OK: $i"
done
