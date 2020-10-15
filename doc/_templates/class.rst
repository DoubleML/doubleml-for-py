{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods %}
   {%- if not item.startswith('_') or item in ['__call__'] %} ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}


{% if methods %}
{% for item in methods %}
{%- if not item.startswith('_') or item in ['__call__'] %}
.. automethod:: {{ name }}.{{ item }}
{% endif %}
{%- endfor %}
{% endif %}