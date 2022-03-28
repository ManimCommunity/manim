{{ name | escape | underline}}

Qualified name: ``{{ fullname | escape }}``

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:
   :private-members:


   {% block methods %}
   {% set methods_to_list = methods if item not in inherited_members %}
   {% set methods_to_list = methods_to_list if item != '__init__' %}
   {%- if methods_to_list %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
      {% for item in methods_to_list %}
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
