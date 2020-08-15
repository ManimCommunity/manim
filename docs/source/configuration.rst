Configuration
=============

Manim provides an extensive configuration system that allows it to adapt to
many different use cases.  The easiest way to do this is through the use of
command line (or *CLI*) arguments.


Command Line Arguments
**********************

All the command line arguments available, as well as the correct way of
executing manim is shown by the command

.. code-block:: bash

   $ manim -h

The output looks as follows.

.. testcode::
   :hide:

   import subprocess
   result = subprocess.run(['manim', '-h'], stdout=subprocess.PIPE)
   print(result.stdout.decode('utf-8'))

.. testoutput::
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

   usage: manim [-h] [-o OUTPUT_FILE] [-p] [-f] [--sound] [--leave_progress_bars]
                [-a] [-w] [-s] [-g] [-i] [--disable_caching] [--flush_cache]
                [--log_to_file] [-c COLOR]
                [--background_opacity BACKGROUND_OPACITY] [--media_dir MEDIA_DIR]
                [--log_dir LOG_DIR] [--tex_template TEX_TEMPLATE] [--dry_run]
                [-t] [-l] [-m] [-e] [-k] [-r RESOLUTION]
                [-n FROM_ANIMATION_NUMBER] [--config_file CONFIG_FILE]
                [-v {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                [--progress_bar True/False]
                {cfg} ... file [scene_names [scene_names ...]]

   Animation engine for explanatory math videos

   positional arguments:
     {cfg}
     file                  path to file holding the python code for the scene
     scene_names           Name of the Scene class you want to see

   optional arguments:
     -h, --help            show this help message and exit
     -o OUTPUT_FILE, --output_file OUTPUT_FILE
                           Specify the name of the output file, if it should be
                           different from the scene class name
     -p, --preview         Automatically open the saved file once its done
     -f, --show_in_file_browser
                           Show the output file in the File Browser
     --sound               Play a success/failure sound
     --leave_progress_bars
                           Leave progress bars displayed in terminal
     -a, --write_all       Write all the scenes from a file
     -w, --write_to_movie  Render the scene as a movie file
     -s, --save_last_frame
                           Save the last frame (and do not save movie)
     -g, --save_pngs       Save each frame as a png
     -i, --save_as_gif     Save the video as gif
     --disable_caching     Disable caching (will generate partial-movie-files
                           anyway).
     --flush_cache         Remove all cached partial-movie-files.
     --log_to_file         Log terminal output to file.
     -c COLOR, --color COLOR
                           Background color
     --background_opacity BACKGROUND_OPACITY
                           Background opacity
     --media_dir MEDIA_DIR
                           directory to write media
     --log_dir LOG_DIR     directory to write log files to
     --tex_template TEX_TEMPLATE
                           Specify a custom TeX template file
     --dry_run             Do a dry run (render scenes but generate no output
                           files)
     -t, --transparent     Render to a movie file with an alpha channel
     -l, --low_quality     Render at low quality (for fastest rendering)
     -m, --medium_quality  Render at medium quality (for much faster rendering)
     -e, --high_quality    Render at high quality (for slightly faster rendering)
     -k, --fourk_quality   Render at 4K quality (slower rendering)
     -r RESOLUTION, --resolution RESOLUTION
                           Resolution, passed as "height,width"
     -n FROM_ANIMATION_NUMBER, --from_animation_number FROM_ANIMATION_NUMBER
                           Start rendering not from the first animation, butfrom
                           another, specified by its index. If you passin two
                           comma separated values, e.g. "3,6", it will endthe
                           rendering at the second value
     --config_file CONFIG_FILE
                           Specify the configuration file
     -v {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --verbosity {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                           Verbosity level. Also changes the ffmpeg log level
                           unless the latter is specified in the config
     --progress_bar True/False
                           Display the progress bar

   Made with <3 by the manim community devs
