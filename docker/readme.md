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
