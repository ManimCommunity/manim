# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.14
ARG UV_VERSION=0.13
# TeX Live is pinned to the frozen final state of a past release ("tlnet-final"
# in the historic archive), so the installation is identical on every build and
# does not depend on the CTAN mirror rotation.
ARG TL_YEAR=2025

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ── Stage 1: TeX Live ────────────────────────────────────────────────────────
# Independent of the Python base image and of the manim sources, so the build
# cache for this (network-heavy) stage survives most changes.
FROM debian:trixie-slim AS texlive

ARG TL_YEAR
# Tried in order; the first repository that completes the installation wins.
ARG TL_REPOSITORIES="\
https://ftp.tu-chemnitz.de/pub/tug/historic/systems/texlive/${TL_YEAR}/tlnet-final \
https://mirrors.tuna.tsinghua.edu.cn/tex-historic-archive/systems/texlive/${TL_YEAR}/tlnet-final"

RUN apt-get -o Acquire::Retries=5 update -qq \
    && apt-get -o Acquire::Retries=5 install --no-install-recommends -y \
        ca-certificates \
        curl \
        libfontconfig1 \
        perl \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

COPY docker/texlive-profile.txt docker/texlive-packages.txt /tmp/

# bin/linux is an architecture-independent link to bin/{x86_64,aarch64}-linux.
RUN set -eu; \
    packages="$(grep -v -e '^#' -e '^[[:space:]]*$' /tmp/texlive-packages.txt)"; \
    for repo in ${TL_REPOSITORIES}; do \
        echo "==> Installing TeX Live ${TL_YEAR} from ${repo}"; \
        rm -rf /tmp/install-tl /usr/local/texlive; \
        mkdir -p /tmp/install-tl; \
        if curl -fsSL --retry 5 --retry-all-errors --connect-timeout 20 --max-time 300 \
                -o /tmp/install-tl-unx.tar.gz "${repo}/install-tl-unx.tar.gz" \
            && tar -xzf /tmp/install-tl-unx.tar.gz -C /tmp/install-tl --strip-components=1 \
            && timeout 30m /tmp/install-tl/install-tl \
                --profile=/tmp/texlive-profile.txt --repository "${repo}" \
            && timeout 30m /usr/local/texlive/bin/*-linux/tlmgr \
                --repository "${repo}" install ${packages}; \
        then \
            installed=1; \
            break; \
        fi; \
        echo "==> Installation from ${repo} failed, trying the next repository" >&2; \
    done; \
    test "${installed:-0}" = 1 || { echo "==> All TeX Live repositories failed" >&2; exit 1; }; \
    ln -s "$(basename /usr/local/texlive/bin/*-linux)" /usr/local/texlive/bin/linux; \
    rm -rf /tmp/install-tl /tmp/install-tl-unx.tar.gz \
        /usr/local/texlive/install-tl.log /usr/local/texlive/texmf-var/web2c/*.log

ENV PATH=/usr/local/texlive/bin/linux:$PATH
RUN latex --version | head -n 1 && dvisvgm --version

# ── Stage 2: Python builder ──────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

RUN apt-get -o Acquire::Retries=5 update -qq \
    && apt-get -o Acquire::Retries=5 install --no-install-recommends -y \
        build-essential \
        cmake \
        pkg-config \
        libcairo2-dev \
        libffi-dev \
        libpango1.0-dev \
        libegl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /opt/manim

# Third-party dependencies, pinned by uv.lock. This layer is only rebuilt when
# pyproject.toml or uv.lock change, not on every change to the manim sources.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project --extra jupyterlab

# manim itself (non-editable, so the sources are not needed at runtime).
# pip is added to the virtualenv from the bundled wheel (no download), so that
# derived images can still `pip install` plugins.
COPY . /opt/manim
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra jupyterlab \
    && /opt/venv/bin/python -m ensurepip \
    && /opt/venv/bin/manim --version

# ── Stage 3: runtime ─────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim

# Runtime libs only:
#  - no ffmpeg: PyAV (av package) bundles its own ffmpeg libraries in av.libs/
#  - OpenGL: keep EGL for headless rendering and libGL as required by moderngl/glcontext
#  - fonts-noto-core instead of fonts-noto (drops CJK noto fonts)
RUN apt-get -o Acquire::Retries=5 update -qq \
    && apt-get -o Acquire::Retries=5 install --no-install-recommends -y \
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
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f

ARG NB_USER=manimuser
ARG NB_UID=1000
ENV USER=${NB_USER} \
    NB_UID=${NB_UID} \
    HOME=/manim

RUN adduser --disabled-password \
        --gecos "Default user" \
        --uid ${NB_UID} \
        ${NB_USER} \
    && mkdir -p ${HOME} \
    && chown -R ${NB_USER}:${NB_USER} ${HOME} \
    && chmod 777 ${HOME}

COPY --from=texlive /usr/local/texlive /usr/local/texlive
COPY --from=builder /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/texlive/bin/linux:$PATH

WORKDIR ${HOME}
USER ${NB_USER}

CMD ["/bin/bash"]
