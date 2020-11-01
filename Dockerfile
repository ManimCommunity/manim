FROM python:3.8
# based on debian

RUN apt update \
    && apt upgrade -y \
    && apt install -y \
        libcairo2-dev \
        ffmpeg \
        texlive \
        texlive-latex-extra \
        texlive-fonts-extra \
        texlive-latex-recommended \
        texlive-science \
        tipa


# choose one of the following installation types:

##
## Option 1:
## Simple installation
##
#RUN pip3 install manimce

##
## Option 2:
## git clone installation
##
#RUN git clone https://github.com/ManimCommunity/manim.git \
#   && cd manim \
#   && python3 -m pip install .
#WORKDIR /manim

##
## Option 3:
## copy local git folder
##
#COPY . /manim
#RUN cd /manim \
#    && python3 -m pip install .
#WORKDIR /manim

##
## Option 4:
## only install manim dependencies
##   and later, mount (using -v) your git folder
##
RUN pip3 install Pillow progressbar grpcio-tools grpcio pydub watchdog tqdm rich numpy pygments scipy colour pycairo
## the same ugly patch, only here there's no need to uninstall
RUN pip3 install --no-binary :all: -U cairocffi --no-cache
RUN pip3 install --no-binary :all: -U pangocffi --no-cache
RUN pip3 install --no-binary :all: -U pangocairocffi --no-cache
## /manim doesn't exist yet in this option. You have to mount it using `docker run -v /your/path/to/manim:/manim`
WORKDIR /manim



## an ugly fix, as of 2020/10/31
## based on:
## https://www.reddit.com/r/manim/comments/jfvcya/weekly_help_thread_ask_for_manim_help_here/
## which linked to:
## https://github.com/leifgehrmann/pangocairocffi/pull/13#issuecomment-714570325
##
## fix only if using one of the first three options
#RUN python3 -m pip uninstall -y pangocairocffi cairocffi pangocffi
#RUN pip3 install --no-binary :all: -U cairocffi --no-cache
#RUN pip3 install --no-binary :all: -U pangocffi --no-cache
#RUN pip3 install --no-binary :all: -U pangocairocffi --no-cache


ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/bin/bash"]


# build with:
# docker build --label="manimce" --tag="manimce" .

# test option 1 with:
# docker run --rm -it manimce "apt install -y wget && wget 'https://raw.githubusercontent.com/ManimCommunity/manim/master/example_scenes/basic.py' && python -m manim basic.py SquareToCircle --low_quality -s && echo yay || echo nay"

# test the other options with:
# docker run --rm -it manimce "python -m manim example_scenes/basic.py SquareToCircle --low_quality -s > /dev/null 2>&1 && echo yay || echo nay"
