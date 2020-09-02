
from docutils.parsers.rst import directives, Directive
from docutils.parsers.rst.directives.images import Image

import jinja2
import os
from os.path import relpath

import shutil


class ManimDirective(Directive):
    r"""Implementation of a ``.. manim::`` directive.

    """
    has_content = True
    required_arguments = 1
    optional_arguments = 1
    final_argument_whitespace = False

    def run(self):
        print(self.arguments)
        clsname = self.arguments[0]
        include_source = True
        if len(self.arguments) == 2:
            include_source = self.arguments[1] == "True"

        state_machine = self.state_machine
        document = state_machine.document

        source_file_name = document.attributes['source']
        source_rel_name = relpath(source_file_name, setup.confdir)
        source_rel_dir = os.path.dirname(source_rel_name)
        while source_rel_dir.startswith(os.path.sep):
            source_rel_dir = source_rel_dir[1:]

        dest_dir = os.path.abspath(os.path.join(setup.app.builder.outdir,
                                            source_rel_dir))
        print(dest_dir)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        source_block = ['.. code-block:: python', '', 
                        *['    ' + line for line in self.content]]
        source_block = '\n'.join(source_block)

        rendered_template = jinja2.Template(TEMPLATE).render(
            include_source=include_source,
            source_block=source_block,
            clsname=clsname
        )
        state_machine.insert_input(rendered_template.split('\n'), 
                                   source=document.attributes['source'])
        exec('\n'.join(self.content) + f'\n\n{clsname}()', globals())
        # copy video file?
        destvid = os.path.join(dest_dir, f'{clsname}.mp4')
        shutil.copyfile(f'media/videos/1080p60/{clsname}.mp4', destvid)
        return []



def setup(app):
    import manim
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir
    app.add_directive('manim', ManimDirective)

    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata


TEMPLATE = r"""
{% if include_source %}
{{ source_block }}
{% endif %}

.. raw:: html

    <video style="width: 100%;" controls src="./{{ clsname }}.mp4"></video>

"""

