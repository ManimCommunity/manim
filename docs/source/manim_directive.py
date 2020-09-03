
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
    optional_arguments = 0
    option_spec = {
        'display_source': bool,
        'save_as_gif': bool,
        'save_last_frame': bool,
    }
    final_argument_whitespace = True

    def run(self):
        clsname = self.arguments[0]

        display_source = self.options.get('display_source', False)
        save_as_gif = self.options.get('save_as_gif', False)
        save_last_frame = self.options.get('save_last_frame', False)
        assert not (save_as_gif and save_last_frame)

        state_machine = self.state_machine
        document = state_machine.document

        source_file_name = document.attributes['source']
        source_rel_name = relpath(source_file_name, setup.confdir)
        source_rel_dir = os.path.dirname(source_rel_name)
        while source_rel_dir.startswith(os.path.sep):
            source_rel_dir = source_rel_dir[1:]
    
        source_dir = os.path.abspath(os.path.join(setup.app.builder.srcdir,
                                                  source_rel_dir))

        dest_dir = os.path.abspath(os.path.join(setup.app.builder.outdir,
                                            source_rel_dir))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        source_block = ['.. code-block:: python', '', 
                        *['    ' + line for line in self.content]]
        source_block = '\n'.join(source_block)
        
        file_writer_config_code = [
            'file_writer_config["media_dir"] = "./source/media"',
            'file_writer_config["images_dir"] = "./source/media/images"',
            'file_writer_config["video_dir"] = "./source/media/videos"',
            f'file_writer_config["save_last_frame"] = {save_last_frame}',
            f'file_writer_config["save_as_gif"] = {save_as_gif}'
        ]
        code = [
            'from manim import *', 
            *file_writer_config_code, 
            *self.content, 
            f'{clsname}()'
        ]
        exec('\n'.join(code), globals())

        # copy video file to output directory
        if not (save_as_gif or save_last_frame):
            filename = f'{clsname}.mp4'
            filesrc = f'source/media/videos/1080p60/{filename}'
            destfile = os.path.join(dest_dir, filename)
            shutil.copyfile(filesrc, destfile)
        elif save_as_gif:
            filename = f'{clsname}.gif'
            filesrc = f'source/media/videos/1080p60/{filename}'
        elif save_last_frame:
            filename = f'{clsname}.png'
            filesrc = f'source/media/images/{clsname}.png'
        else:
            raise ValueError('Invalid combination of render flags received.')

        rendered_template = jinja2.Template(TEMPLATE).render(
            clsname=clsname,
            display_source=display_source,
            filesrc=filesrc[6:],
            save_last_frame=save_last_frame,
            save_as_gif=save_as_gif,
            source_block=source_block,
        )
        state_machine.insert_input(rendered_template.split('\n'), 
                                   source=document.attributes['source'])

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
{% if display_source %}
{{ source_block }}
{% endif %}

{% if not (save_as_gif or save_last_frame) %}
.. raw:: html

    <video style="width: 100%;" controls loop src="./{{ clsname }}.mp4"></video>
{% elif save_as_gif %}
.. image:: {{ filesrc }}
    :align: center
{% elif save_last_frame %}
.. image:: {{ filesrc }}
    :align: center
{% endif %}
"""

