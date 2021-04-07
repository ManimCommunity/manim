#!/bin/bash

# MIT License

# Copyright (c) 2020 Federico Carboni, 2021 The Manim Community Developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

echo "Downloading Pango"

pango_url='https://download.gnome.org/sources/pango/1.48/pango-1.48.4.tar.xz'
darwin_name='pango-darwin'
darwin_temp="/tmp/$darwin_name"
darwin_archive="$darwin_name.tar.gz"

download() {
    echo "Pango: downloading $1 to $2"
    curl -L $1 -o $2
}

create_archive() {
    echo "Pango: creating archive $1 from $2"
    curdir=$PWD
    cd $2
    tar -czvf "$curdir/$1" *
    cd $curdir
}

tar_extract() {
    echo "Pango: extracting $2 from $1 to $3"
    tar -xf $1 --wildcards -O $2 > $3
}

remove() {
    echo "Pango: removing $1"
}

create_darwin_archive() {
    echo 'Pango: creating darwin archive'
    download $pango_url "/tmp/$darwin_name.7z"
    mkdir -p $darwin_temp
    7z e "/tmp/$darwin_name.7z" "-o$darwin_temp" 'pango'
    remove "/tmp/$darwin_name.7z"
    create_archive $darwin_archive $darwin_temp
    remove $darwin_temp
    echo 'Pango: darwin archive created successfully'
}

create_darwin_archive

# Expose paths to the archives
echo "::set-output name=darwin-pango-path::$darwin_archive"