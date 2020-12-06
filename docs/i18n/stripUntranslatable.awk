BEGIN {
state=-1
precomment=""
msgidstr=""
sharpedmsgidstr=""
msgstrstr=""
sharpedmsgstrstr=""
acceptable=1
untranslatablefile="./untranslatable.po"
}

$0~/^$/ {
	if (state<=0){
		if(acceptable==1){
			print precomment
		}else{
			print "#Detected untranslatable text:"
			print precomment
		}
		precomment=""
	}else{
		if(acceptable==1){
			print precomment
			print msgidstr
			print msgstrstr
		}else{
			print "#Detected untranslatable text:"
			print precomment
			print sharpedmsgidstr
			print sharpedmsgstrstr
			print precomment>>untranslatablefile
			print msgidstr>>untranslatablefile
			print msgstrstr>>untranslatablefile
		}
		print ""
		state=-1
		acceptable=1
		precomment=""
		msgidstr=""
		msgstrstr=""
	}
}
$0~/^#/ {
	precomment=(state==-1)?$0:precomment"\n"$0
	state=0
}
$0~/^msgid/ {
	state=1
	msgidstr=$0
	sharpedmsgidstr="#"$0
}
$0~/^\"/ {
	if(state==1){
		msgidstr=msgidstr"\n"$0
		sharpedmsgidstr=sharpedmsgidstr"\n#"$0
	}else{
		msgstrstr=msgstrstr"\n"$0
		sharpedmsgstrstr=sharpedmsgstrstr"\n#"$0
	}
}
$0~/^msgstr/ {
	state=2
	msgstrstr=$0
	sharpedmsgstrstr="#"$0
}

$0~/^msgid ":ref:`[a-zA-Z]*`"/ {
	acceptable=0
}
$0~/^"/ {	
	acceptable=1
}
END {
print ""
}
