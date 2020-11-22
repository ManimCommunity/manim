{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:

   {% block methods %}
   {%- if methods %}

   {%- if objname in custom_method_order %}
   {%- set sections = custom_method_order[objname] %}
   {%- for section in sections %}

   .. rubric:: {{ section }} {{ _('Methods') }}
   .. autosummary::
      :nosignatures:

     {% for item in sections[section] if item != '__init__' and item not in inherited_members %}
      ~{{ name }}.{{ item }}
     {%- endfor %}

   {%- endfor %}

   {%- else %}
   .. rubric:: {{ _('Methods') }}
   .. autosummary::
      :nosignatures:

      {% for item in methods if item != '__init__' and item not in inherited_members %}
       ~{{ name }}.{{ item }}
      {%- endfor %}

   {%- endif %}

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
