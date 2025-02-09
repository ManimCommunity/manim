# Installing Manim locally via pipx (on Arch BTW™)

Arch, and maybe other Linux distributions, prevent installing via `pip` either
globally or at a user level scope.

<details><summary>Example: `pip install <package>` error</summary>
```
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try 'pacman -S
    python-xyz', where xyz is the package you are trying to
    install.

    If you wish to install a non-Arch-packaged Python package,
    create a virtual environment using 'python -m venv path/to/venv'.
    Then use path/to/venv/bin/python and path/to/venv/bin/pip.

    If you wish to install a non-Arch packaged Python application,
    it may be easiest to use 'pipx install xyz', which will manage a
    virtual environment for you. Make sure you have python-pipx
    installed via pacman.

note: If you believe this is a mistake, please contact your Python installation
      or OS distribution provider. You can override this, at the risk of
      breaking your Python installation or OS, by passing
        --break-system-packages.
hint: See PEP 668 for the detailed specification.
```
</details>

## Setup for managing multiple Python runtimes

> Note: this sub-section may be ignored if **not** using any transcription or
> voice-recording plugins.

Install `pyenv` to manage Python run-times without hosing the host;

- Arch
   ```bash
   sudo pacman -S pyenv
   ```

> Note: `sox` and `gettext` be dependencies for using Manim Community
> transcription/voice-recording plugins, and you may wish to install both at
> this time via your preferred system-level package manager too.

- Activate the `pyenv` stuff
   ```bash
   pyenv init >> ~/.bashrc
   source ~/.bashrc
   ```

Install Python version 3.9.9

```bash
pyenv install 3.9.9
```

> Note: as of 2025-02-03 the above version is the highest
> [open-whisper](https://github.com/openai/whisper?tab=readme-ov-file#setup)
> supports.


## Aktually install Manim Community

Install Manim Community via PipX using above Python runtime version

```bash
pipx install manim --python 3.9.9
```

> Note: again, as of 2025-02-03 the above version is the highest
> [open-whisper](https://github.com/openai/whisper?tab=readme-ov-file#setup)
> supports.  If **not** using transcription or voice-recording plugins, then
> you may choose to remove the `--python <version>` portion from above command.

## Inject additional dependencies

### Inject voice-over tools into the `manim` virtual environment

```bash
pipx inject manim setuptools

pipx inject manim TTS

pipx inject manim "manim-voiceover[coqui,gtts,recorder,transcribe]"
```

<details><summary>Note: about Manim Voiceover injection</summary>
- `TTS` is needed by `conqui`, but not in the package's requirements text file?!
- `gtts` uses Google API, so is just as likely to be murdered as any other Google project...  But it do work, for now.
- `recorder` doesn't seem to work on Arch Linux without explicitly setting `transcription_model=None`, ex.
   ```python
   class MeSuperAwesomeTalky(VoiceoverScene):
       def construct(self):
           self.set_speech_service(RecorderService(transcription_model=None))
   ```
</details>

<details><summary>Warning: adjusting `manim-voiceover` installed features requires re-install</summary>
```bash
pipx uninject manim manim-voiceover

pipx inject manim "manim-voiceover[all]"
```
</details>

### Inject `importlib_metadata` or `metadata`

This may only be necessary if, again, you intent to utilize transcription
and/or voice-over features.

```bash
pipx inject manim importlib_metadata
```

<details><summary>Warning: if/when OpenAI Whisper starts using Python version 3.10 or greater</summary>
...  use the following instead of `importlib_metadata`

```bash
pipx inject manim metadata
```

For the curious, the above `importlib_metadata` vs `metadata` dependency
injection _should_ solve errors similar to;

```
Traceback (most recent call last):
  File "~/.local/bin/manim", line 5, in <module>
    from manim.__main__ import main
  File "~/.local/share/pipx/venvs/manim/lib/python3.9/site-packages/manim/__init__.py", line 112, in <module>
    from .plugins import *
  File "~/.local/share/pipx/venvs/manim/lib/python3.9/site-packages/manim/plugins/__init__.py", line 4, in <module>
    from manim.plugins.plugins_flags import get_plugins, list_plugins
  File "~/.local/share/pipx/venvs/manim/lib/python3.9/site-packages/manim/plugins/plugins_flags.py", line 9, in <module>
    from importlib_metadata import entry_points
ModuleNotFoundError: No module named 'importlib_metadata'
```
</details>

### Inject interactive Python `ipython`

```bash
pipx inject manim ipython
```

<details><summary>Example: usage and error this resolves</summary>
**Error message**

```
ModuleNotFoundError: No module named 'IPython'
```

**CLI command to produce error**

```bash
manim -pql <FILE> <SCENE> --renderer=opengl --enable_gui
```
</details>

## IDE integration with pipx managed Manim

### Vim YouCompleteMe `.ycm_extra_conf.py`

```python
#!/usr/bin/env python

def Settings( **kwargs ):
    return {
        'interpreter_path': '~/.local/share/pipx/venvs/manim/bin/python'
    }
```
