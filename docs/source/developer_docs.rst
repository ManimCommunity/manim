#######################
Developer Documentation
#######################

The goal of this page is to give developers interested in contributing
to manim a deeper look into its inner workings and hopefully make it
easier to understand the codebase. The target audience is therefore
those interested in developing manim and may not be as useful to users.
If you fall into the latter category and want to learn how to create
animations with manim we have documentation for that
`here <https://docs.manim.community/en/stable/>`__

Rendering logic
===============

CLI
===

OpenGLRenderer and Shaders
==========================

The Opengl rendering pipeline in manim involves logic in both python
code and programs called
`shaders <https://en.wikipedia.org/wiki/Shader>`__. Shaders are programs
designed to run on graphics cards. They are written in a language called
glsl that is similar in syntax to c++. Manim uses
`moderngl <https://github.com/moderngl/moderngl#:~:text=ModernGL%20is%20a%20python%20wrapper,requires%20a%20steep%20learning%20curve.>`__,
a python library that acts as a wrapper around opengl.

Shaders can be broken into three categories `Vertex
Shaders <#Vertex-Shaders>`__, `Geometry Shaders <#Geometry-Shaders>`__
and `Fragment Shaders <#Fragment-Shaders>`__. Generally speaking each
``OpenGLMobject`` will be assigned to a vertex shader, a fragment shader
and optionally a geometry shader. In some cases multiple of a given type
of shader can also be used.

The order of processing is **vertex shader** -> **geometry shader** ->
**fragment shader**.

Assigning Manim Shaders
-----------------------

In this section we will discuss how manim’s existing shaders are
assigned. There are also options to use `custom
shaders <#Using-Custom-Shaders>`__.

Manim’s shaders are stored in
`manim/renderer/shaders <https://github.com/ManimCommunity/manim/tree/main/manim/renderer/shaders>`__.
You will notice that shaders are stored in groups containing at least a
vertex shader, a fragment shader and optionally a geometry shader. This
makes up the opengl pipeline. In manim we only need to point it to the
directory and it will detect the shaders based on the following naming
convention:

-  Vertex shader -> vert.glsl
-  Geometry shader -> geom.glsl
-  Fragment shader -> frag.glsl

Manim knows what shader folder to look by class attributes. There are
two base classes that are relevant here:

-  ``OpenGLMobject`` uses a single class attribute ``shader_folder`` to
   define the shader that should be used. By default no shader is
   defined, however subclasses that require a shader should set this. An
   example of a class that defines this attribute is ``OpenGLPMobject``
   as shown below

.. code:: py

   class OpenGLPMobject(OpenGLMobject):
       shader_folder = "true_dot"

-  ``OpenGLVMobject`` uses two groups of shaders, one for stroke and one
   for fills and so two class attributes can be set to define the
   shaders that are to be used. These attributes are
   ``stroke_shader_folder`` that defaults to ``quadratic_bezier_stroke``
   and ``fill_shader_folder`` that defaults to
   ``quadratic_bezier_fill``. When extending this class these attributes
   can be set for subclasses to use different shaders for example the
   below class will extend ``OpenGLVMobject`` and set its own shaders

.. code:: py

   class MyCustomClass(OpenGLVMobject):
       stroke_shader_folder = "vectorized_mobject_stroke"
       fill_shader_folder = "vectorized_mobject_fill"

Rendering Flow
--------------

The entry point for the opengl rendering flow is in the ``render``
method in the ``OpenglRenderer``. This is called for each time step from
the ``Scene`` object. This will call ``update_frame`` and cycle through
each ``OpenGLMObject`` in the scene, rendering each one in the
``render_mobject`` method. Objects may be assigned different shaders and
so each object will have its own ``ShaderWrapper``. This is a container
that holds what it needs to render that given object such as the name of
the folder containing the shader, the data to be passed to the shader
etc. Most of the ``render_mobject`` involves preprocessing such as
updating data before the opengl stage. After this preprocessing stage
the ``Mesh`` object’s ``render`` method is called. This method contains
the main logic bridging manim and opengl. It takes the data that has
been created with manim and passes it to moderngl with vertex buffers
and vertex arrays.

Passing Data to Shaders
-----------------------

We need to pass data to the shaders to be processed and rendered on the
graphics card. There are different types of data that can be used by
shaders, the first type we will discuss is data that can vary for each
vertex, in opengl these are known as **attributes**.

Attributes
~~~~~~~~~~

If we take a simple triangle as an example, it can be defined by 3
vertices. We need a way to pass data such as color and position of each
vertex. We do this using a flexible
`descriptor <https://docs.python.org/3.8/howto/descriptor.html>`__
``_Data`` that can be found
`here <https://github.com/ManimCommunity/manim/blob/main/manim/utils/config_ops.py>`__.
This allows us to use keys such as ‘points’ to hold our position data
and map them to the shader input ‘point’ attribute for a each vertex.
Let’s look at an example of how this is set up.

Taking ``OpenGLMobject`` as an example we initialise ``points`` as at
the class level as below:

.. code:: py

   class OpenGLMobject:
       ...
       points = _Data()

This will create the key in our descriptor and we can now treat it like
an instance attribute, and in this case this assign positions to each
vertex as below.

.. code:: py

   self.points = points  # numpy array containing xyz points with shape (n, 3)

Now that we have our attribute created and all our points are ready,
however this array isn’t passed directly to the vertex shader. The
vertex shader may use different keys, for example it may have an input
such as ``in vec3 point;``. The reason for this is that the vertex
shader will only take in a single vertex (point) at a time, so if we
have three vertices for our triangle the vertex shader will only have
access to one at a time. ``OpenGLMObject`` contains a method
``read_data_to_shader`` that will map from manim’s data keys to the
shaders keys, in other words map the ‘points’ key to the vertex shader’s
‘point’ key.

After some processing the actual calls to a moderngl context happens in
the ``Mesh`` object’s ``render`` method.

.. code:: py

   vertex_buffer_object = self.shader.context.buffer(shader_attributes.tobytes())

This creates a buffer with the data that originated in the ``_Data``
descriptor that is passed to the shader.

Uniforms
~~~~~~~~

The next type of data we can pass to shaders are **uniforms**. Unlike
attributes, uniforms are not set per vertex, instead they are constant
over a single render. Therefore the data in a uniform will be constant
over a single draw call. This is not the same as an actual constant
however, a real constant will be the same across all draw calls. Taking
a triangle with three vertices again as an example, the uniform will not
change as these vertices are rendered, however we can update the
uniforms each time the whole triangle is rendered.

Uniforms are set in a similar way to attributes - by a descriptor called
``_Uniforms``. An example of initialising a uniform can be seen below:

.. code:: py

   class OpenGLMobject:
       ...
       is_fixed_in_frame = _Uniforms()
       gloss = _Uniforms()
       shadow = _Uniforms()

They can they be set as normal python attributes ``self.gloss=0.0``. As
you can see uniforms follow a similar pattern to attributes. Where they
diverge is how they are passed to the shaders. Unlike attributes they
are set in the ``Mesh`` object’s ``set_uniforms`` method. Each moderngl
context has a program and uniform’s are set using dict-like syntax
``self.shader_program[name] = value``. Taking the gloss uniform as an
example in the shader we will have:

.. code:: glsl

   uniform float gloss;

Manim’s shader class would be assigning this by
``self.shader_program['gloss'] = 0.0``

Using Custom Shaders
--------------------

Vertex Shaders
--------------

Geometry Shaders
----------------

Fragment Shaders
----------------

Testing logic
=============

Graphing in Manim
=================

When working with graphs in Manim, you will mainly be relying on these
files: ``coordianate_systems.py``, ``number_line.py`` and optionally,
``functions.py``/``scale.py``.

Description of graphing classes.
--------------------------------

`CoordinateSystem <https://docs.manim.community/en/latest/reference/manim.mobject.coordinate_systems.CoordinateSystem.html>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``CoordinateSystem`` is the parent class of ``Axes``. It initializes the
following information:
``x_range``/``y_range``/``x_length``/``y_length``. It’s an abstract
class and stores attributes/methods that are meant to be shared among
all graphing-related classes. Although, all current classes inherit from
``Axes`` instead, so there isn’t a clear need for this class. Mostly
exists for organization purposes

`Axes <https://docs.manim.community/en/latest/reference/manim.mobject.coordinate_systems.Axes.html>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Axes`` is the primary graphing class in Manim. Its job is to create
the axes via ``_create_axis`` (which creates ``NumberLine`` and
positions them appropriately). It’s the class from which the methods
defined in ``CoordinateSystem`` are used. ``Axes`` offers many useful
methods, such as specifying points along the graph (``coords_to_point``)
and plotting functions (``plot``). It’s important to remember that the
axes of an ``Axes`` mobject are ``NumberLine``\ s. Therefore, you can
use any methods defined in ``NumberLine`` when dealing with them via
``Axes.x_axis.<method>``, for example.

`NumberLine <https://docs.manim.community/en/latest/reference/manim.mobject.number_line.NumberLine.html>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``NumberLine`` is the true backbone of Manim’s graphing infrastructure.
It handles everything from creating the ticks, labels, lines and
configuration for an axis. It supports different scales (logarithmic,
linear) and its methods account for these. Almost everything it
generates can be accessed after creation and modified. It inherits from
``Line`` to create the actual ``NumberLine``.

Here is an outline of its key methods:

-  ``get_tick_range``: Generates the a list of the position of the
   ticks. So, with an ``x_range`` of ``[1, 10, 2]``, this method would
   generate five evenly spaced ticks. These ticks values are then
   adjusted depending on the scaling via
   ``NumberLine.scaling.function``, but remain evenly spaced due to
   ``number_to_point`` accounting for this scaling.
-  ``get_tick``. Generates a ``Line`` mobject that acts as a tick.
   Called on by ``add_ticks`` (which then calls on ``get_tick_range``).
-  ``get_number_mobject``: Accepts an ``x-value`` and generates a
   ``DecimalNumber`` mob for the number and positions it with
   ``number_to_point``. This is how the numbers are generated by
   ``add_numbers``.
-  ``add_numbers``: Calls on ``get_tick_range`` and iterates through the
   range while calling on ``get_number_mobject`` to generate the
   numbers.
-  ``number_to_point``: The main tool for putting things on the
   NumberLine. It interpolates between the min/max values of the line
   and determines where a specific “x-value” belongs. Accounts for the
   scaling in ``get_tick_range`` via
   ``NumberLine.scaling.inverse_function``.

So, here’s how a ``NumberLine`` is made: Make the line (length
determined by ``x_range``/``length``) -> ``get_tick_range`` determines
where the ticks/numbers will be -> ``add_ticks``/``get_tick`` define the
ticks -> ``get_number_mobject``/``add_numbers`` add the numbers and
``number_to_point`` determines where everything should be placed.

Scaling classes /``add_labels`` can generate custom label mobjects, but
the general idea remains the same.

\_ScaleBase
~~~~~~~~~~~

``_ScaleBase`` is an abstract base class which allows more configuration
for ``NumberLine``/``ParametricFunction``. For example, ``LogBase``
allows a user to define a custom base and offers the
``get_custom_labels`` method for generating labels for a ``NumberLine``
in the form of base^exponent.

Each scaling class must define a ``function`` (used in
``get_tick_range``/``ParametricFunction``) and an ``inverse_function``,
which is simply the inverse of ``function`` and is used when plotting on
a ``NumberLine``.

For instance, the ``function`` for ``LogBase`` is simply an exponential
in the form of ``base^{value}``, whereas the inverse function is the
logarithmic function with the same base. A custom rule for generating
labels on the graph can optionally be defined for use with graphing. See
the ``get_custom_labels`` implementation in ``LogBase`` for inspiration.

**NOTE**: ``function``/``inverse_function`` may not be valid everywhere
in the domain, e.g. log(0) is undefined.

`NumberPlane <https://docs.manim.community/en/latest/reference/manim.mobject.coordinate_systems.NumberPlane.html?highlight=NumberPlane>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apart from some minor stylistic changes, ``NumberPlane`` is effectively
equivalent to ``Axes`` in terms of generating the axes. However, the
background lines introduce more complexity to this class.

Here’s the breakdown for the background lines:
``_init_background_lines`` –> ``_get_lines`` –>
``_get_lines_parallel_to_axis``.

-  ``_init_background_lines``: Applies the styling defined for the
   background lines and calls ``_get_lines``.
-  ``_get_lines``: An intermediary method which passes in parameters
   into ``_get_lines_parallel_to_axis``. It calls the method twice, once
   to generate the horizontal lines, and once for the vertical lines. It
   puts these lines into two separate ``VGroups`` and returns them to
   ``_init_background_lines``.
-  ``_get_lines_parallel_to_axis``: Actually generates the lines. It
   iterates over the ``x_step`` of the perpendicular axis:
   ``y_range[2]`` for the horizontal lines and ``x_range[2]`` for the
   vertical lines. There are some precautions taken when ``0`` is not
   included in the range and depending on the scaling function of the
   axis. 🚧More description needed🚧

ThreeDAxes
~~~~~~~~~~

``ThreeDAxes`` is really just an ``Axes`` that creates a third
``NumberLine`` mobject and rotates it to position it along the z-axis.
Its only unique method is ``get_z_axis_label``.

PolarPlane
~~~~~~~~~~

🚧Under construction🚧

ComplexPlane
~~~~~~~~~~~~

A ``NumberPlane`` which has support for complex numbers.

ParametricFunction
~~~~~~~~~~~~~~~~~~

🚧Under construction🚧

Key methods
-----------

🚧Under Construction🚧
