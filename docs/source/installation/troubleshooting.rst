Troubleshooting
===============

This page contains solutions to common issues you might encounter when installing or using Manim.

.. _dvisvgm-pdf-issue:

``dvisvgm`` does not support PDF to SVG conversion (macOS)
--------------------------------------------------------

If you are on macOS and see an error like ``unknown option --pdf`` when Manim tries to render LaTeX, it usually means your ``dvisvgm`` tool is missing Ghostscript support.

**Why does this happen?**

When you install MacTeX via Homebrew, ``dvisvgm`` is installed but Ghostscript (a required dependency for PDF to SVG conversion) is not included by default.

**How to fix it**

1. Install Ghostscript using Homebrew:

   .. code-block:: bash

      brew install ghostscript

2. If the issue persists, you may need to set the ``LIBGS`` environment variable:

   .. code-block:: bash

      export LIBGS=$(brew --prefix)/lib/libgs.dylib

3. Verify that the fix worked:

   .. code-block:: bash

      dvisvgm -l

   You should see ``ps    dvips PostScript specials`` in the output.

For more details, see the `dvisvgm FAQ <https://dvisvgm.de/FAQ/>`_.

Still having issues?
--------------------

If you encounter other problems, please:
- Check the `FAQ <https://docs.manim.community/en/stable/faq/index.html>`_
- Search existing `GitHub Issues <https://github.com/ManimCommunity/manim/issues>`_
- Ask for help on the `Discord community <https://discord.com/invite/bYCyhM9Kz2>`_
