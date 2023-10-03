import moderngl
import numpy as np
import re

from enum import Enum


class BufferLayout(Enum):
    PACKED = 0
    STD140 = 1

class BufferFormat:

    def __init__(
        self,
        *,
        name: str,
        shape: tuple[int, ...]
    ) -> None:
        super().__init__()
        self._name_ = name
        self._shape_ = shape

    @staticmethod
    def _name_() -> str:
        return ""

    @staticmethod
    def _shape_() -> tuple[int, ...]:
        return ()

    @staticmethod
    def _itemsize_() -> int:
        # Implemented in subclasses.
        return 0

    @staticmethod
    def _size_(
        shape: tuple[int, ...]
    ) -> int:
        return int(np.prod(shape, dtype=np.int32))

    @staticmethod
    def _nbytes_(
        itemsize: int,
        size: int
    ) -> int:
        return itemsize * size

    @staticmethod
    def _is_empty_(
        size: int
    ) -> bool:
        return not size

    @staticmethod
    def _dtype_() -> np.dtype:
        # Implemented in subclasses.
        return np.dtype("f4")

    @staticmethod
    def _pointers_() -> tuple[tuple[tuple[str, ...], int], ...]:
        # Implemented in subclasses.
        return ()

    def _get_np_buffer_and_pointers(self) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, int]]]:

        def get_np_buffer_pointer(
            np_buffer: np.ndarray,
            name_chain: list[str]
        ) -> np.ndarray:
            if not name_chain:
                return np_buffer["_"]
            name = name_chain.pop(0)
            return get_np_buffer_pointer(np_buffer[name], name_chain)

        np_buffer = np.zeros(self._shape_, dtype=self._dtype_)
        np_buffer_pointers = {
            ".".join(name_chain): (get_np_buffer_pointer(np_buffer, list(name_chain)), base_ndim)
            for name_chain, base_ndim in self._pointers_
        }
        return np_buffer, np_buffer_pointers

    def _write(
        self,
        data_dict: dict[str, np.ndarray]
    ) -> bytes:
        np_buffer, np_buffer_pointers = self._get_np_buffer_and_pointers()
        for key, (np_buffer_pointer, base_ndim) in np_buffer_pointers.items():
            data = data_dict[key]
            if not np_buffer_pointer.size:
                assert not data.size
                continue
            data_expanded = np.expand_dims(data, axis=tuple(range(-2, -base_ndim)))
            assert np_buffer_pointer.shape == data_expanded.shape
            np_buffer_pointer[...] = data_expanded
        return np_buffer.tobytes()

    def _read(
        self,
        data_bytes: bytes
    ) -> dict[str, np.ndarray]:
        data_dict: dict[str, np.ndarray] = {}
        np_buffer, np_buffer_pointers = self._get_np_buffer_and_pointers()
        np_buffer[...] = np.frombuffer(data_bytes, dtype=np_buffer.dtype).reshape(np_buffer.shape)
        for key, (np_buffer_pointer, base_ndim) in np_buffer_pointers.items():
            data_expanded = np_buffer_pointer[...]
            data = np.squeeze(data_expanded, axis=tuple(range(-2, -base_ndim)))
            data_dict[key] = data
        return data_dict

class StructuredBufferFormat(BufferFormat):

    def __init__(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        children: list[BufferFormat],
        layout: BufferLayout
    ) -> None:
        structured_base_alignment = 16
        offsets: list[int] = []
        offset: int = 0
        for child in children:
            if layout == BufferLayout.STD140:
                if isinstance(child, StructuredBufferFormat):
                    base_alignment = structured_base_alignment
                else:
                    raise TypeError
                offset += (-offset) % base_alignment
            offsets.append(offset)
            offset += child._nbytes_
        if layout == BufferLayout.STD140:
            offset += (-offset) % structured_base_alignment

        super().__init__(
            name=name,
            shape=shape
        )
        self._children_ = tuple(children)
        self._offsets_ = tuple(offsets)
        self._itemsize_ = offset

    @staticmethod
    def _children_() -> tuple[BufferFormat, ...]:
        return ()

    @staticmethod
    def _offsets_() -> tuple[int, ...]:
        return ()

    @staticmethod
    def _dtype_(
        children__name: tuple[str, ...],
        children__dtype: tuple[np.dtype, ...],
        children__shape: tuple[tuple[int, ...], ...],
        offsets: tuple[int, ...],
        itemsize: int
    ) -> np.dtype:
        return np.dtype({
            "names": children__name,
            "formats": list(zip(children__dtype, children__shape, strict=True)),
            "offsets": list(offsets),
            "itemsize": itemsize
        })

    @staticmethod
    def _pointers_(
        children__name: tuple[str, ...],
        children__pointers: tuple[tuple[tuple[tuple[str, ...], int], ...], ...]
    ) -> tuple[tuple[tuple[str, ...], int], ...]:
        return tuple(
            ((child_name,) + name_chain, base_ndim)
            for child_name, child_pointers in zip(children__name, children__pointers, strict=True)
            for name_chain, base_ndim in child_pointers
        )

class Buffer:

    def __init__(
        self,
        field: str,
        child_structs: dict[str, list[str]] | None,
        array_lens: dict[str, int] | None
    ) -> None:
        super().__init__()
        self._field_ = field
        if child_structs is not None:
            self._child_struct_items_ = tuple(
                (name, tuple(child_struct_fields))
                for name, child_struct_fields in child_structs.items()
            )
        if array_lens is not None:
            self._array_len_items_ = tuple(array_lens.items())

    @staticmethod
    def _field_() -> str:
        return ""

    @staticmethod
    def _child_struct_items_() -> tuple[tuple[str, tuple[str, ...]], ...]:
        return ()

    @staticmethod
    def _array_len_items_() -> tuple[tuple[str, int], ...]:
        return ()

    @staticmethod
    def _layout_() -> BufferLayout:
        return BufferLayout.PACKED

    @staticmethod
    def _buffer_format_(
        field: str,
        child_struct_items: tuple[tuple[str, tuple[str, ...]], ...],
        array_len_items: tuple[tuple[str, int], ...],
        layout: BufferLayout
    ) -> BufferFormat:

        def parse_field_str(
            field_str: str,
            array_lens_dict: dict[str, int]
        ) -> tuple[str, str, tuple[int, ...]]:
            pattern = re.compile(r"""
                (?P<dtype_str>\w+?)
                \s
                (?P<name>\w+?)
                (?P<shape>(\[\w+?\])*)
            """, flags=re.VERBOSE)
            match_obj = pattern.fullmatch(field_str)
            assert match_obj is not None
            dtype_str = match_obj.group("dtype_str")
            name = match_obj.group("name")
            shape = tuple(
                int(s) if re.match(r"^\d+$", s := index_match.group(1)) is not None else array_lens_dict[s]
                for index_match in re.finditer(r"\[(\w+?)\]", match_obj.group("shape"))
            )
            return (dtype_str, name, shape)

        def get_buffer_format(
            field: str,
            child_structs_dict: dict[str, tuple[str, ...]],
            array_lens_dict: dict[str, int]
        ) -> BufferFormat:
            dtype_str, name, shape = parse_field_str(field, array_lens_dict)
            child_struct_fields = child_structs_dict.get(dtype_str)
            return StructuredBufferFormat(
                
                name=name,
                shape=shape,
                children=[
                    get_buffer_format(
                        child_struct_field,
                        child_structs_dict,
                        array_lens_dict
                    )
                    for child_struct_field in child_struct_fields
                ],
                layout=layout
            )

        return get_buffer_format(
            field,
            dict(child_struct_items),
            dict(array_len_items)
        )

    @staticmethod
    def _buffer_pointer_keys_(
        buffer_format__pointers: tuple[tuple[tuple[str, ...], int], ...]
    ) -> tuple[str, ...]:
        return tuple(".".join(name_chain) for name_chain, _ in buffer_format__pointers)


    @staticmethod
    def _data_dict_() -> dict[str, np.ndarray]:
        return {}

    @staticmethod
    def _buffer_(
        ctx: moderngl.Context,
        data_dict: dict[str, np.ndarray],
        buffer_format: BufferFormat
    ) -> moderngl.Buffer:
        return ctx.buffer(data=buffer_format._write(data_dict))

    def write(
        self,
        data_dict: dict[str, np.ndarray]
    ) -> None:
        self._data_dict_ = data_dict