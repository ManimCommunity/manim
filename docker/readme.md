See the [main README](https://github.com/ManimCommunity/manim/blob/main/README.md) for some instructions on how to use this image.

# Building the image
The docker image corresponding to the checked out version of the git repository
can be built by running
```
docker build -t manimcommunity/manim:TAG -f docker/Dockerfile .
```
from the root directory of the repository.

Multi-platform builds are possible by running
```
docker buildx build --push --platform linux/arm64/v8,linux/amd64 --tag manimcommunity/manim:TAG -f docker/Dockerfile .
```
from the root directory of the repository.

To check a built image, render the verification scene (LaTeX, XeLaTeX, Pango, video encoding):
```
docker run --rm -v "$PWD/docker:/test:ro" manimcommunity/manim:TAG \
    manim render -ql --media_dir /tmp/media /test/verify_image.py VerifyImage
```

# Build notes
- The image is built via a multi-stage Dockerfile: TeX Live, the Python
  virtualenv and the runtime image are separate stages, and build dependencies
  are not carried into the runtime stage.
- TeX Live is pinned to the frozen final state of a past release (`TL_YEAR`)
  from the TeX Live historic archive, with fallback mirrors (`TL_REPOSITORIES`).
  Bump `TL_YEAR` once the next release is frozen. TeX packages are listed in
  `texlive-packages.txt`.
- Python dependencies are installed from `uv.lock`.
- CI builds each platform on a native runner and caches layers, so the TeX
  Live stage is usually not rebuilt.

# Runtime notes
- The image does not include the `ffmpeg` CLI binary.
- The default TeX installation is minimal and does not include `ctex`.
  `tlmgr` installs further packages from the same frozen TeX Live release.
- Headless OpenGL rendering relies on EGL/GL runtime libraries available in the
  image.
