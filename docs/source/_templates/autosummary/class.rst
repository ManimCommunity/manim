{{ name | escape | underline}}

Qualified name: ``{{ fullname | escape }}``

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:
   :private-members:


   {% block methods %}
   {% set no_init_m = methods | select('ne', '__init__') %}
   {%- if set(no_init_m).difference(set(inherited_members)) %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
      {% for item in methods if item != '__init__' and item not in inherited_members %}
      ~{{ name }}.{{ item }}
      {%- endfor %}
   {%- endif %}
   {%- endblock %}

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
     {% for item in attributes %}
     ~{{ name }}.{{ item }}
     {%- endfor %}
   {%- endif %}
   {% endblock %}
