import attr
import functools
import typing

dclass = functools.partial(attr.s, auto_attribs=True, eq=False)

