# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.14-slim AS builder

RUN apt-get update -qq \
    && apt-get install --no-install-recommends -y \
        build-essential \
        gcc \
        cmake \
        make \
        pkg-config \
        wget \
        libcairo2-dev \
        libffi-dev \
        libpango1.0-dev \
        libegl-dev \
    && rm -rf /var/lib/apt/lists/*

# Setup a minimal TeX Live installation (no ctex: drops ~100 MB of CJK fonts/packages)
COPY docker/texlive-profile.txt /tmp/
ENV PATH=/usr/local/texlive/bin/armhf-linux:/usr/local/texlive/bin/aarch64-linux:/usr/local/texlive/bin/x86_64-linux:$PATH
RUN wget -O /tmp/install-tl-unx.tar.gz http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz \
    && mkdir /tmp/install-tl \
    && tar -xzf /tmp/install-tl-unx.tar.gz -C /tmp/install-tl --strip-components=1 \
    && /tmp/install-tl/install-tl --profile=/tmp/texlive-profile.txt \
    && tlmgr install \
        amsmath babel-english cbfonts-fd cm-super count1to doublestroke dvisvgm everysel \
        fontspec frcursive fundus-calligra gnu-freefont jknapltx latex-bin \
        mathastext microtype multitoc physics prelim2e preview ragged2e relsize rsfs \
        setspace standalone tipa wasy wasysym xcolor xetex xkeyval \
    && rm -rf /tmp/install-tl /tmp/install-tl-unx.tar.gz

# Install manim into an isolated virtualenv
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY . /opt/manim
WORKDIR /opt/manim
RUN pip install --no-cache-dir .[jupyterlab]

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.14-slim

# Runtime libs only:
#  - no ffmpeg: PyAV (av package) bundles its own ffmpeg libraries in av.libs/
#  - OpenGL: keep EGL for headless rendering and libGL as required by moderngl/glcontext
#  - fonts-noto-core instead of fonts-noto (drops CJK noto fonts)
RUN apt-get update -qq \
    && apt-get install --no-install-recommends -y \
        libcairo2 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libpangoft2-1.0-0 \
        libffi8 \
        libegl1 \
        libgl1 \
        ghostscript \
        fonts-noto-core \
        fontconfig \
    && rm -rf /var/lib/apt/lists/*

RUN fc-cache -fv

# Copy TeX Live from builder
ENV PATH=/usr/local/texlive/bin/armhf-linux:/usr/local/texlive/bin/aarch64-linux:/usr/local/texlive/bin/x86_64-linux:$PATH
COPY --from=builder /usr/local/texlive /usr/local/texlive

# Copy the pre-built virtualenv from builder
ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder /opt/venv /opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ARG NB_USER=manimuser
ARG NB_UID=1000
ENV USER=${NB_USER}
ENV NB_UID=${NB_UID}
ENV HOME=/manim

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

WORKDIR ${HOME}
RUN chown -R ${NB_USER}:${NB_USER} ${HOME} && chmod 777 ${HOME}
USER ${NB_USER}

CMD ["/bin/bash"]
