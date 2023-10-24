import numpy as np
import math


class STD140BufferFormat:
    _GL_DTYPES: dict[str, tuple[str, int, tuple[int, ...]]] = {
        "int": ("i", np.float32, (1,)),
        "ivec2": ("i", np.float32, (2,)),
        "ivec3": ("i", np.float32, (3,)),
        "ivec4": ("i", np.float32, (4,)),
        "uint": ("u", np.float32, (1,)),
        "uvec2": ("u", np.float32, (2,)),
        "uvec3": ("u", np.float32, (3,)),
        "uvec4": ("u", np.float32, (4,)),
        "float": ("f", np.float32, (1,)),
        "vec2": ("f", np.float32, (2, 1)),
        "vec3": ("f", np.float32, (3, 1)),
        "vec4": ("f", np.float32, (4, 1)),
        "mat2": ("f", np.float32, (2, 2)),
        "mat2x3": ("f", np.float32, (2, 3)),  # TODO: check order
        "mat2x4": ("f", np.float32, (2, 4)),
        "mat3x2": ("f", np.float32, (3, 2)),
        "mat3": ("f", np.float32, (3, 3)),
        "mat3x4": ("f", np.float32, (3, 4)),
        "mat4x2": ("f", np.float32, (4, 2)),
        "mat4x3": ("f", np.float32, (4, 3)),
        "mat4": ("f", np.float32, (4, 4)),
        "double": ("f", np.float64, (1,)),
        "dvec2": ("f", np.float64, (2,)),
        "dvec3": ("f", np.float64, (3,)),
        "dvec4": ("f", np.float64, (4,)),
        "dmat2": ("f", np.float64, (2, 2)),
        "dmat2x3": ("f", np.float64, (2, 3)),
        "dmat2x4": ("f", np.float64, (2, 4)),
        "dmat3x2": ("f", np.float64, (3, 2)),
        "dmat3": ("f", np.float64, (3, 3)),
        "dmat3x4": ("f", np.float64, (3, 4)),
        "dmat4x2": ("f", np.float64, (4, 2)),
        "dmat4x3": ("f", np.float64, (4, 3)),
        "dmat4": ("f", np.float64, (4, 4)),
    }

    def __init__(
        self,
        name: str,
        struct: tuple[(str, str), ...],
    ) -> None:
        self.dtype = []
        self._offsets = dict()
        byte_offset = 0
        for data_type, var_name in struct:
            base_char, base_bytesize, shape = self._GL_DTYPES[data_type]
            shape = dict(enumerate(shape))
            col_len, row_len = shape.get(0, 1), shape.get(1, 1)
            col_padding = 0 if row_len == 1 and (col_len == 1 or col_len == 2) else 4 - col_len
            self._offsets[var_name] = col_padding
            shape = (col_len + col_padding,)
            if row_len > 1:
                shape = (row_len,) + shape
            final_shape = shape
            if byte_offset % 16 != 0 and col_len != 1:
                padding_for_alignment = (((16 - byte_offset) % 16) // 4,)
                self.dtype.append(
                    (f"padding-{byte_offset}", base_bytesize, padding_for_alignment)
                )
                byte_offset += padding_for_alignment[0]*4
            self.dtype.append((var_name, base_bytesize, final_shape))
            byte_offset += math.prod(final_shape + (base_bytesize(0).nbytes,))
        self.data = np.zeros(1, dtype=self.dtype)


    def _write_padded(self, data, var: str) -> np.ndarray:
        """Automatically adds padding to data if necessary. Used by write"""
        try:
            return np.pad(data, ((0, 0), (0, self._offsets[var])), mode="constant")
        except:
            return np.pad(data, ((0, self._offsets[var])), mode="constant")

    def write(self, data: dict) -> None:
        for key, val in data.items():

            # print("WRITING", key,val)
            self.data[key] = self._write_padded(val, key)
