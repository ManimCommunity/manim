{{ name | escape | underline}}

Qualified name: ``{{ fullname | escape }}``

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:
   :private-members:


   {% block methods %}
   {% set displayed_methods = [] %}
   {%- for m in methods %}
      {%- if m != '__init__' %}
         {%- if m not in inherited_members %}
            {% set displayed_methods = displayed_methods + [m] %}
         {%- endif %}
      {%- endif %}
   {%- endfor %}
   {%- if displayed_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
      {% for item in displayed_methods %}
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
