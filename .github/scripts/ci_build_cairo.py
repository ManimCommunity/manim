# Logic is as follows:
# 1. Download cairo source code: https://cairographics.org/releases/cairo-<version>.tar.xz
# 2. Verify the downloaded file using the sha256sums file: https://cairographics.org/releases/cairo-<version>.tar.xz.sha256sum
# 3. Extract the downloaded file.
# 4. Create a virtual environment and install meson and ninja.
# 5. Run meson build in the extracted directory. Also, set required prefix.
# 6. Run meson compile -C build.
# 7. Run meson install -C build.

import hashlib
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
import typing
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from sys import stdout

CAIRO_VERSION = "1.18.0"
CAIRO_URL = f"https://cairographics.org/releases/cairo-{CAIRO_VERSION}.tar.xz"
CAIRO_SHA256_URL = f"{CAIRO_URL}.sha256sum"

VENV_NAME = "meson-venv"
BUILD_DIR = "build"
INSTALL_PREFIX = Path(__file__).parent.parent.parent / "third_party" / "cairo"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def is_ci():
    return os.getenv("CI", None) is not None


def download_file(url, path):
    logger.info(f"Downloading {url} to {path}")
    block_size = 1024 * 1024
    with urllib.request.urlopen(url) as response, open(path, "wb") as file:
        while True:
            data = response.read(block_size)
            if not data:
                break
            file.write(data)


def verify_sha256sum(path, sha256sum):
    with open(path, "rb") as file:
        file_hash = hashlib.sha256(file.read()).hexdigest()
    if file_hash != sha256sum:
        raise Exception("SHA256SUM does not match")


def extract_tar_xz(path, directory):
    with tarfile.open(path) as file:
        file.extractall(directory)


def run_command(command, cwd=None, env=None):
    process = subprocess.Popen(command, cwd=cwd, env=env)
    process.communicate()
    if process.returncode != 0:
        raise Exception("Command failed")


@contextmanager
def gha_group(title: str) -> typing.Generator:
    if not is_ci():
        yield
        return
    print(f"\n::group::{title}")
    stdout.flush()
    try:
        yield
    finally:
        print("::endgroup::")
        stdout.flush()


def set_env_var_gha(name: str, value: str) -> None:
    if not is_ci():
        return
    env_file = os.getenv("GITHUB_ENV", None)
    if env_file is None:
        return
    with open(env_file, "a") as file:
        file.write(f"{name}={value}\n")
    stdout.flush()


def get_ld_library_path(prefix: Path) -> str:
    # given a prefix, the ld library path can be found at
    # <prefix>/lib/* or sometimes just <prefix>/lib
    # this function returns the path to the ld library path

    # first, check if the ld library path exists at <prefix>/lib/*
    ld_library_paths = list(prefix.glob("lib/*"))
    if len(ld_library_paths) == 1:
        return ld_library_paths[0].absolute().as_posix()

    # if the ld library path does not exist at <prefix>/lib/*,
    # return <prefix>/lib
    ld_library_path = prefix / "lib"
    if ld_library_path.exists():
        return ld_library_path.absolute().as_posix()
    return ""


def main():
    if sys.platform == "win32":
        logger.info("Skipping build on windows")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        with gha_group("Downloading and Extracting Cairo"):
            logger.info(f"Downloading cairo version {CAIRO_VERSION}")
            download_file(CAIRO_URL, os.path.join(tmpdir, "cairo.tar.xz"))

            logger.info("Downloading cairo sha256sum")
            download_file(CAIRO_SHA256_URL, os.path.join(tmpdir, "cairo.sha256sum"))

            logger.info("Verifying cairo sha256sum")
            with open(os.path.join(tmpdir, "cairo.sha256sum")) as file:
                sha256sum = file.read().split()[0]
            verify_sha256sum(os.path.join(tmpdir, "cairo.tar.xz"), sha256sum)

            logger.info("Extracting cairo")
            extract_tar_xz(os.path.join(tmpdir, "cairo.tar.xz"), tmpdir)

        with gha_group("Installing meson and ninja"):
            logger.info("Creating virtual environment")
            run_command([sys.executable, "-m", "venv", os.path.join(tmpdir, VENV_NAME)])

            logger.info("Installing meson and ninja")
            run_command(
                [
                    os.path.join(tmpdir, VENV_NAME, "bin", "pip"),
                    "install",
                    "meson",
                    "ninja",
                ]
            )

        env_vars = {
            # add the venv bin directory to PATH so that meson can find ninja
            "PATH": f"{os.path.join(tmpdir, VENV_NAME, 'bin')}{os.pathsep}{os.environ['PATH']}",
        }

        with gha_group("Building and Installing Cairo"):
            logger.info("Running meson setup")
            run_command(
                [
                    os.path.join(tmpdir, VENV_NAME, "bin", "meson"),
                    "setup",
                    BUILD_DIR,
                    f"--prefix={INSTALL_PREFIX.absolute().as_posix()}",
                    "--buildtype=release",
                    "-Dtests=disabled",
                ],
                cwd=os.path.join(tmpdir, f"cairo-{CAIRO_VERSION}"),
                env=env_vars,
            )

            logger.info("Running meson compile")
            run_command(
                [
                    os.path.join(tmpdir, VENV_NAME, "bin", "meson"),
                    "compile",
                    "-C",
                    BUILD_DIR,
                ],
                cwd=os.path.join(tmpdir, f"cairo-{CAIRO_VERSION}"),
                env=env_vars,
            )

            logger.info("Running meson install")
            run_command(
                [
                    os.path.join(tmpdir, VENV_NAME, "bin", "meson"),
                    "install",
                    "-C",
                    BUILD_DIR,
                ],
                cwd=os.path.join(tmpdir, f"cairo-{CAIRO_VERSION}"),
                env=env_vars,
            )

        logger.info(f"Successfully built cairo and installed it to {INSTALL_PREFIX}")


if __name__ == "__main__":
    if "--set-env-vars" in sys.argv:
        with gha_group("Setting environment variables"):
            # append the pkgconfig directory to PKG_CONFIG_PATH
            set_env_var_gha(
                "PKG_CONFIG_PATH",
                f"{Path(get_ld_library_path(INSTALL_PREFIX), 'pkgconfig').as_posix()}{os.pathsep}"
                f'{os.getenv("PKG_CONFIG_PATH", "")}',
            )
            set_env_var_gha(
                "LD_LIBRARY_PATH",
                f"{get_ld_library_path(INSTALL_PREFIX)}{os.pathsep}"
                f'{os.getenv("LD_LIBRARY_PATH", "")}',
            )
        sys.exit(0)
    main()
