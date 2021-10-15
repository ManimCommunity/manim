#!/bin/bash
rm -f translatable.po
rm -f untranslatable.po
pot_dir_prefix="gettext/"
echo ""
for srcFile in `find $pot_dir_prefix -name "*.pot" -type f`
do
	printf "\r\033[KCleaning file \e[32m$srcFile\e[0m"
	dstFile="$srcFile.bld"
	awk -f stripUntranslatable.awk $srcFile | sed '/POT-Creation-Date/d'> $dstFile
	cat $dstFile >> translatable.po
	mv $dstFile $srcFile
done
echo ""
