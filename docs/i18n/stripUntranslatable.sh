#!/bin/bash
rm -f translatable.po
rm -f untranslatable.po
rm -rf build-en
mkdir build-en
pot_dir_prefix="en/LC_MESSAGES/"
for i in `find $pot_dir_prefix -name "*.po"`
do
	srcFile=$i
	destFile="build-$srcFile"
	mkdir -p `dirname $destFile`
	awk -f stripUntranslatable.awk $srcFile > $destFile
	cat $destFile >> translatable.po
	echo "OK: $srcFile"
done
rm -rf en
mv build-en en
