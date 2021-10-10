#!/bin/bash
rm -f translatable.po
rm -f untranslatable.po
pot_dir_prefix="gettext/"
for srcFile in `find $pot_dir_prefix -name "*.pot"`
do
	dstFile="$i.bld"
	awk -f stripUntranslatable.awk $srcFile > $dstFile
	cat $dstFile >> translatable.po
	mv $srcFile $dstFile
	rm $dstFile
	echo "OK: $srcFile"
done
