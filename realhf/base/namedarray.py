from typing import Callable, Dict, List, Tuple
import ast
import copy
import enum
import hashlib
import itertools
import pickle
import time
import types
import warnings

import numpy as np
import torch

import realhf.base.logging as logging

sparse_tensor_fields = []


def dense_tensor_size(x: torch.Tensor):
    assert not x.is_sparse
    return x.numel() * x.element_size()


def sparse_tensor_size(x: torch.Tensor):
    assert x.is_sparse
    indices_size = x.indices().numel() * x.indices().element_size()
    values_size = x.values().numel() * x.values().element_size()
    return indices_size + values_size


class NamedArrayLoadingError(Exception):
    pass


class NamedArrayEncodingMethod(bytes, enum.Enum):
    """Encoding protocol of a NamedArray object.

    Principle: compress huge non-random data and pickle small data.

    TL;DR:
    + InferenceStream
        ++ If your observation include images, use PICKLE_COMPRESS.
        ++ Otherwise, use PICKLE or PICKLE_DICT.
    + SampleStream:
        ++ If your observation include images, use OBS_COMPRESS or COMPRESS_EXPECT_POLICY_STATE.
        ++ Otherwise,
            +++ if the amount of data is huge (say, multi-agent envs), use COMPRESS_EXPECT_POLICY_STATE.
            +++ otherwise, use PICKLE or PICKLE_DICT.
    """

    PICKLE_DICT = b"0001"  # Convert NamedArray to dict, then pickle.
    PICKLE = b"0002"  # Directly pickle.
    RAW_BYTES = b"0003"  # Send raw bytes of flattened numpy arrays.
    RAW_COMPRESS = b"0004"  # Send compressed bytes of flattened numpy arrays.
    COMPRESS_PICKLE = (
        b"0005"  # Convert numpy array to compressed bytes, then pickle.
    )
    PICKLE_COMPRESS = b"0006"  # Pickle, then compress pickled bytes.
    OBS_COMPRESS = b"0007"  # Convert flattened numpy array to bytes and only compress observation.
    COMPRESS_EXCEPT_POLICY_STATE = b"0008"  # Compress all bytes except for policy states, which are basically random numbers.
    TENSOR_COMPRESS = b"0009"  # Turn certain tensors into sparse tensors, then pickle. Turn other tensors into numpy arrays then compress.


logger = logging.getLogger("NamedArray")


def _namedarray_op(op):

    def fn(self, value):
        if not (
            isinstance(value, NamedArray)  # Check for matching structure.
            and getattr(value, "_fields", None) == self._fields
        ):
            if not isinstance(value, NamedArray):
                # Repeat value for each but respect any None.
                value = tuple(None if s is None else value for s in self)
            else:
                raise ValueError(
                    "namedarray - set an item with a different data structure"
                )
        try:
            xs = {}
            for j, ((k, s), v) in enumerate(zip(self.items(), value)):
                if s is not None and v is not None:
                    exec(f"xs[k] = (s {op} v)")
                else:
                    exec(f"xs[k] = None")
        except (ValueError, IndexError, TypeError) as e:
            print(s.shape, v.shape)
            raise Exception(
                f"{type(e).__name__} occured in {self.__class__.__name__}"
                " at field "
                f"'{self._fields[j]}': {e}"
            ) from e
        return NamedArray(**xs)

    return fn


def _namedarray_iop(iop):

    def fn(self, value):
        if not (
            isinstance(value, NamedArray)  # Check for matching structure.
            and getattr(value, "_fields", None) == self._fields
        ):
            if not isinstance(value, NamedArray):
                # Repeat value for each but respect any None.
                value = {
                    k: None if s is None else value for k, s in self.items()
                }
            else:
                raise ValueError(
                    "namedarray - set an item with a different data structure"
                )
        try:
            for j, (k, v) in enumerate(zip(self.keys(), value.values())):
                if self[k] is not None and v is not None:
                    exec(f"self[k] {iop} v")
        except (ValueError, IndexError, TypeError) as e:
            raise Exception(
                f"{type(e).__name__} occured in {self.__class__.__name__}"
                " at field "
                f"'{self._fields[j]}': {e}"
            ) from e
        return self

    return fn


def _numpy_dtype_to_str(dtype):
    if dtype == np.uint8:
        return "uint8"
    elif dtype == bool:
        return "bool"
    elif dtype == np.float32:
        return "float32"
    elif dtype == np.float16:
        return "float16"
    elif dtype == np.float64:
        return "float64"
    elif dtype == np.int32:
        return "int32"
    elif dtype == np.int64:
        return "int64"
    elif dtype == np.dtype("<U7"):
        return "<U7"
    else:
        raise NotImplementedError(
            f"Data type to string not implemented: {dtype}."
        )


def dumps(namedarray_obj, method="pickle_dict"):
    """Serialize a NamedArray object to bytes.

    Args:
        namedarray_obj (NamedArray): The NamedArray object to be serialized.
        method (str): The serialization method.
    """

    if "compress" in method:
        try:
            import blosc
        except ModuleNotFoundError:

            class blosc:

                def compress(x, *args, **kwargs):
                    return x

            warnings.warn(
                "Module `blosc` not found in the image. Abort NamedArray compression."
            )

    def compress_large(buf, typesize=4, cname="lz4"):
        chunk_size = 2**30
        num_chunks = len(buf) // chunk_size + 1
        chunks = [
            blosc.compress(
                buf[i * chunk_size : (i + 1) * chunk_size],
                typesize=typesize,
                cname=cname,
            )
            for i in range(num_chunks)
        ]
        # logger.info(f"Compress {len(buf)} bytes to {sum([len(chunk) for chunk in chunks])} bytes, num_chunks = {num_chunks}")
        return num_chunks, chunks

    def _namedarray_to_bytes_list(
        x, compress: bool, compress_condition: Callable[[str], bool]
    ):
        flattened_entries = flatten(x)
        flattened_bytes = []
        for k, v in flattened_entries:
            k_ = k.encode("ascii")
            dtype_ = v_ = shape_ = b""
            if v is not None:
                dtype_ = _numpy_dtype_to_str(v.dtype).encode("ascii")
                v_ = v.tobytes()
                shape_ = str(tuple(v.shape)).encode("ascii")
                if compress and compress_condition(k):
                    v_ = blosc.compress(v_, typesize=4, cname="lz4")
            flattened_bytes.append((k_, dtype_, shape_, v_))
        return list(itertools.chain.from_iterable(flattened_bytes))

    def _tensor_namedarray_to_bytes_list(x):
        t0 = time.monotonic()
        flattened_entries = flatten(x)
        flattened_bytes = []
        for k, v in flattened_entries:
            k_ = k.encode("ascii")
            dtype_ = v_ = shape_ = b""
            if k in sparse_tensor_fields:
                t1 = time.monotonic()
                assert v is not None, f"Sparse tensor field {k} cannot be None."
                logger.info(
                    f"Dense tensor {k} size {dense_tensor_size(v)} bytes"
                )
                vshape = v.shape
                v = v.to_sparse()
                logger.info(
                    f"Sparse tensor {k} size {sparse_tensor_size(v)} bytes"
                )
                v_ = pickle.dumps(v)
                logger.info(
                    f"dump sparse tensor {k} time {time.monotonic() - t1:4f}, v shape {vshape}"
                )
            elif v is not None:
                v = v.cpu().numpy()
                dtype_ = _numpy_dtype_to_str(v.dtype).encode("ascii")
                v_ = v.tobytes()
                shape_ = str(tuple(v.shape)).encode("ascii")
                v_ = blosc.compress(v_, typesize=4, cname="lz4")
            flattened_bytes.append((k_, dtype_, shape_, v_))
        # logger.info(f"dump namedarray time {time.monotonic() - t0:4f}")
        return list(itertools.chain.from_iterable(flattened_bytes))

    if method == "pickle_dict":
        bytes_list = [
            NamedArrayEncodingMethod.PICKLE_DICT.value,
            pickle.dumps(
                (namedarray_obj.__class__.__name__, namedarray_obj.to_dict())
            ),
        ]
    elif method == "pickle":
        bytes_list = [
            NamedArrayEncodingMethod.PICKLE.value,
            pickle.dumps(namedarray_obj),
        ]
    elif method == "raw_bytes":
        bytes_list = [
            NamedArrayEncodingMethod.RAW_BYTES.value
        ] + _namedarray_to_bytes_list(namedarray_obj, False, lambda x: False)
    elif method == "raw_compress":
        bytes_list = [
            NamedArrayEncodingMethod.RAW_COMPRESS.value
        ] + _namedarray_to_bytes_list(namedarray_obj, True, lambda x: True)
    elif method == "compress_pickle":
        bytes_list = [
            NamedArrayEncodingMethod.COMPRESS_PICKLE.value,
            pickle.dumps(
                _namedarray_to_bytes_list(namedarray_obj, True, lambda x: True)
            ),
        ]
    elif method == "pickle_compress":
        # bytes_list = [
        #     NamedArrayEncodingMethod.PICKLE_COMPRESS.value,
        #     blosc.compress(pickle.dumps(namedarray_obj), typesize=4, cname='lz4')
        # ]
        bytes_list = [NamedArrayEncodingMethod.PICKLE_COMPRESS.value]
        buf = pickle.dumps(namedarray_obj)
        num_chunks, chunks = compress_large(buf)
        bytes_list += chunks
    elif method == "obs_compress":
        bytes_list = [
            NamedArrayEncodingMethod.OBS_COMPRESS.value
        ] + _namedarray_to_bytes_list(
            namedarray_obj, True, lambda x: ("obs" in x)
        )
    elif method == "compress_except_policy_state":
        bytes_list = [
            NamedArrayEncodingMethod.COMPRESS_EXCEPT_POLICY_STATE.value
        ] + _namedarray_to_bytes_list(
            namedarray_obj, True, lambda x: ("policy_state" not in x)
        )

    elif method == "tensor_compress":
        bytes_list = [
            NamedArrayEncodingMethod.TENSOR_COMPRESS.value
        ] + _tensor_namedarray_to_bytes_list(namedarray_obj)
    else:
        raise NotImplementedError(
            f"Unknown method {method}. Available are {[m.name.lower() for m in NamedArrayEncodingMethod]}."
        )

    return bytes_list + [pickle.dumps(dict(**namedarray_obj.metadata))]


def loads(b):
    # safe import
    if b[0] in [b"0004", b"0004", b"0005", b"0006", b"0007", b"0008", b"0009"]:
        try:
            import blosc
        except ModuleNotFoundError:

            class blosc:

                def decompress(x, *args, **kwargs):
                    return x

    def _parse_namedarray_from_bytes_list(
        xs, compressed: bool, compress_condition: Callable[[str], int]
    ):
        flattened = []
        for i in range(len(xs) // 4):
            k = xs[4 * i].decode("ascii")
            if xs[4 * i + 1] != b"":
                buf = xs[4 * i + 3]
                if compressed and compress_condition(k):
                    buf = blosc.decompress(buf)
                v = np.frombuffer(
                    buf, dtype=np.dtype(xs[4 * i + 1].decode("ascii"))
                ).reshape(*ast.literal_eval(xs[4 * i + 2].decode("ascii")))
            else:
                v = None
            flattened.append((k, v))
        return from_flattened(flattened)

    def _parse_tensor_namedarray_from_bytes_list(xs):
        t0 = time.monotonic()
        flattened = []
        for i in range(len(xs) // 4):
            k = xs[4 * i].decode("ascii")
            if xs[4 * i + 3] == b"":
                # None
                v = None
            elif xs[4 * i + 1] == b"":
                t1 = time.monotonic()
                # sparse tensor
                v = pickle.loads(xs[4 * i + 3])
                assert (
                    torch.is_tensor(v) and v.is_sparse
                ), f"Field {k} is not a sparse tensor, but is serialized as one."
                logger.info(
                    f"Sparse tensor {k} size {sparse_tensor_size(v)} bytes"
                )
                v = v.to_dense()
                logger.info(
                    f"Dense tensor {k} size {dense_tensor_size(v)} bytes"
                )
                logger.info(
                    f"load sparse tensor {k} time {time.monotonic() - t1:4f}, v shape {v.shape}"
                )
            else:
                # dense tensor
                buf = xs[4 * i + 3]
                buf = blosc.decompress(buf)
                v = np.frombuffer(
                    buf, dtype=np.dtype(xs[4 * i + 1].decode("ascii"))
                ).reshape(*ast.literal_eval(xs[4 * i + 2].decode("ascii")))
                v = torch.from_numpy(v)
            flattened.append((k, v))
        # logger.info(f"load namedarray time {time.monotonic() - t0:4f}")
        return from_flattened(flattened)

    if b[0] == NamedArrayEncodingMethod.PICKLE_DICT.value:
        class_name, values = pickle.loads(b[1])
        namedarray_obj = from_dict(values=values)
    elif b[0] == NamedArrayEncodingMethod.PICKLE.value:
        namedarray_obj = pickle.loads(b[1])
    elif b[0] == NamedArrayEncodingMethod.RAW_BYTES.value:
        namedarray_obj = _parse_namedarray_from_bytes_list(
            b[1:-1], False, lambda x: False
        )
    elif b[0] == NamedArrayEncodingMethod.RAW_COMPRESS.value:
        namedarray_obj = _parse_namedarray_from_bytes_list(
            b[1:-1], True, lambda x: True
        )
    elif b[0] == NamedArrayEncodingMethod.COMPRESS_PICKLE.value:
        namedarray_obj = _parse_namedarray_from_bytes_list(
            pickle.loads(b[1]), True, lambda x: True
        )
    elif b[0] == NamedArrayEncodingMethod.PICKLE_COMPRESS.value:
        # namedarray_obj = pickle.loads(blosc.decompress(b[1]))
        chunks = b[1:-1]
        buf = b"".join([blosc.decompress(chunk) for chunk in chunks])
        namedarray_obj = pickle.loads(buf)
    elif b[0] == NamedArrayEncodingMethod.OBS_COMPRESS.value:
        namedarray_obj = _parse_namedarray_from_bytes_list(
            b[1:-1], True, lambda x: ("obs" in x)
        )
    elif b[0] == NamedArrayEncodingMethod.COMPRESS_EXCEPT_POLICY_STATE.value:
        namedarray_obj = _parse_namedarray_from_bytes_list(
            b[1:-1], True, lambda x: ("policy_state" not in x)
        )
    elif b[0] == NamedArrayEncodingMethod.TENSOR_COMPRESS.value:
        namedarray_obj = _parse_tensor_namedarray_from_bytes_list(b[1:-1])
    else:
        raise NotImplementedError(
            f"Unknown NamedArrayEncodingMethod value {b[:4]}. "
            f"Existing are {[m for m in NamedArrayEncodingMethod]}."
        )

    namedarray_obj.clear_metadata()
    metadata = pickle.loads(b[-1])
    namedarray_obj.register_metadata(**metadata)

    return namedarray_obj


class NamedArray:
    """A class acting like a dict that holds torch tensors as values.

    NamedArray supports dict-like unpacking and string indexing, and exposes integer slicing reads
    and writes applied to all contained objects, which must share
    indexing (__getitem__) behavior (e.g. numpy arrays or torch tensors).

    NamedArray also supports metadata, which is useful for recording sequence lengths.

    Example:

    >>> class Point(NamedArray):
    ...     def __init__(self,
    ...         x: np.ndarray,
    ...         y: np.ndarray,
    ...         ):
    ...         super().__init__(x=x, y=y)
    >>> p=Point(np.array([1,2]), np.array([3,4]))
    >>> p
    Point(x=array([1, 2]), y=array([3, 4]))
    >>> p[:-1]
    Point(x=array([1]), y=array([3]))
    >>> p[0]
    Point(x=1, y=3)
    >>> p.x
    array([1, 2])
    >>> p['y']
    array([3, 4])
    >>> p[0] = 0
    >>> p
    Point(x=array([0, 2]), y=array([0, 4]))
    >>> p[0] = Point(5, 5)
    >>> p
    Point(x=array([5, 2]), y=array([5, 4]))
    >>> 'x' in p
    True
    >>> list(p.keys())
    ['x', 'y']
    >>> list(p.values())
    [array([5, 2]), array([5, 4])]
    >>> for k, v in p.items():
    ...     print(k, v)
    ...
    x [5 2]
    y [5 4]
    >>> def foo(x, y):
    ...     print(x, y)
    ...
    >>> foo(**p)
    [5 2] [5 4]
    """

    _reserved_slots = ["_NamedArray__metadata", "_fields"]

    def __init__(self, **kwargs):
        self._fields = list(sorted(kwargs.keys()))
        self.__metadata = types.MappingProxyType({})
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def metadata(self):
        """Return the metadata of the NamedArray object.

        :return: A dict-like metadata of the NamedArray object.
        :rtype: MappingProxyType
        """
        return self.__metadata

    def register_metadata(self, **kwargs):
        """Register metadata to the NamedArray object.

        :param kwargs: Metadata to be registered.
        :type kwargs: Dict
        """
        for k in self._fields:
            if k in kwargs.keys():
                raise KeyError(
                    "Keys of metadata should be different from data fields!"
                )

        self.__metadata = types.MappingProxyType({**self.__metadata, **kwargs})

    def pop_metadata(self, key):
        """Clear a single metadata entry named "key".

        :param key: The key to be removed.
        :type key: str
        :return: The value of the removed key.
        :rtype: Any
        """
        metadatadict = dict(self.__metadata)
        value = metadatadict.pop(key)
        self.__metadata = types.MappingProxyType(metadatadict)
        return value

    def clear_metadata(self):
        """Clear all metadata recorded in this object."""
        self.__metadata = types.MappingProxyType({})

    def __iter__(self):
        for k in self._fields:
            yield getattr(self, k)

    def __setattr__(self, loc, value):
        """Set attributes in a `NamedArray` object.

        Unknown fields cannot be created.

        :param loc: The attribute name to be set.
        :type loc: str or slice
        :param value: The value to be set.
        :type value: Any
        """
        if not (loc in NamedArray._reserved_slots or loc in self._fields):
            self._fields.append(loc)
        super().__setattr__(loc, value)

    def __getitem__(self, loc):
        """If the index is string, return getattr(self, index).
        If the index is integer/slice, return a new dataclass instance containing
        the selected index or slice from each field.

        :param loc: Key or indices to get.
        :type loc: str or slice
        :return: An element of NamedArray or a new NamedArray
            object composed of the slices.
        :rtype: NamedArray or Any
        """
        if isinstance(loc, str):
            # str indexing like in dict
            return getattr(self, loc)
        else:
            sliced_namedarray = dict()
            try:
                for s in self._fields:
                    if self[s] is None:
                        sliced_namedarray[s] = None
                    else:
                        sliced_namedarray[s] = self[s][loc]
            except IndexError as e:
                raise Exception(
                    f"IndexError occured when slicing `NamedArray`."
                    f"Field {s} with shape {self[s].shape} and slice {loc}."
                ) from e
            return self.__class__(
                **{
                    s: None if self[s] is None else self[s][loc]
                    for s in self._fields
                }
            )

    def __setitem__(self, loc, value):
        """If input value is the same dataclass type, iterate through its
        fields and assign values into selected index or slice of corresponding
        field.  Else, assign whole of value to selected index or slice of
        all fields. Ignore fields that are both None.

        :param loc: Key or indices to set.
        :type loc: str or slice
        :param value: An NamedArray instance with the same structure
            or elements of the NamedArray object.
        :type value: Any
        """
        if isinstance(loc, str):
            if loc not in self._fields:
                self._fields.append(loc)
            setattr(self, loc, value)
        else:
            if not (
                isinstance(value, NamedArray)  # Check for matching structure.
                and getattr(value, "_fields", None) == self._fields
            ):
                if not isinstance(value, NamedArray):
                    # Repeat value for each but respect any None.
                    value = tuple(None if s is None else value for s in self)
                else:
                    raise ValueError(
                        "namedarray - set an item with a different data structure"
                    )
            try:
                for j, (s, v) in enumerate(zip(self, value)):
                    if s is not None and v is not None:
                        s[loc] = v
            except (ValueError, IndexError, TypeError) as e:
                raise Exception(
                    f"Error occured occured in {self.__class__.__name__} when assigning value"
                    " at field "
                    f"'{self._fields[j]}': {e}"
                ) from e

    def __contains__(self, key):
        """Checks presence of a field name (unlike tuple; like dict).

        :param key: The queried field name.
        :type key: str
        :return: Query result.
        :rtype: bool
        """
        return key in self._fields

    def __getstate__(self):
        return {
            "__metadata": dict(**self.metadata),
            **{k: v for k, v in self.items()},
        }

    def __setstate__(self, state):
        self.__init__(**{k: v for k, v in state.items() if k != "__metadata"})
        if state["__metadata"] is not None:
            self.clear_metadata()
            self.register_metadata(**state["__metadata"])

    def values(self):
        for v in self:
            yield v

    def keys(self):
        for k in self._fields:
            yield k

    def __len__(self):
        return len(self._fields)

    def length(self, dim=0):
        for k, v in self.items():
            if v is None:
                continue
            elif isinstance(v, (np.ndarray, torch.Tensor)):
                if dim < v.ndim:
                    return v.shape[dim]
                else:
                    continue
            else:
                continue
        else:
            raise IndexError(f"No entry has shape on dim={dim}.")

    def unique_of(self, field, exclude_values=(None,)):
        unique_values = np.unique(self[field])
        unique_values = unique_values[
            np.in1d(unique_values, exclude_values, invert=True)
        ]
        if len(unique_values) != 1:
            return None
        else:
            return unique_values[0]

    def items(self):
        """Iterate over ordered (field_name, value) pairs like a dict."""
        for k, v in zip(self._fields, self):
            yield k, v

    def to_dict(self):
        result = {}
        for k, v in self.items():
            if isinstance(v, NamedArray):
                result[k] = v.to_dict()
            elif v is None:
                result[k] = None
            else:
                result[k] = v
        return result

    @property
    def shape(self):
        return recursive_apply(self, lambda x: x.shape).to_dict()

    def size(self):
        return self.shape

    def __str__(self):
        return f"{self.__class__.__name__}({', '.join(k+'='+repr(v) for k, v in self.items())})"

    def __hash__(self):
        d = self.to_dict()
        to_hash = []
        for k, v in d.items():
            if torch.is_tensor(v):
                vv = int(hashlib.md5(v.numpy().tobytes()).hexdigest(), 16)
            else:
                vv = hash(v)
            to_hash.append(k)
            to_hash.append(vv)
        serialized = pickle.dumps(to_hash)
        return int(hashlib.md5(serialized).hexdigest(), 16)

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        metadata = {}
        for k, v in self.__dict__.items():
            if isinstance(v, types.MappingProxyType):
                metadata = copy.deepcopy(dict(v), memo)
                continue
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        result.clear_metadata()
        result.register_metadata(**metadata)
        return result

    __add__ = _namedarray_op("+")
    __sub__ = _namedarray_op("-")
    __mul__ = _namedarray_op("*")
    __truediv__ = _namedarray_op("/")
    __iadd__ = _namedarray_iop("+=")
    __isub__ = _namedarray_iop("-=")
    __imul__ = _namedarray_iop("*=")
    __itruediv__ = _namedarray_iop("/=")
    __repr__ = __str__


def from_dict(values: Dict):
    """Create namedarray object from Nested Dict of arrays.

    Example:
    >>> a = from_dict({"x": np.array([1, 2]), "y": np.array([3,4])})
    >>> a.x
    array([1, 2])
    >>> a.y
    array([3, 4])
    >>> a[1:]
    NamedArray(x=[2],y=[4])
    >>> obs = {"state":{"speed": np.array([1, 2, 3]), "position": np.array([4, 5])}, "vision": np.array([[7],[8],[9]])}
    >>> obs_na = from_dict(obs)
    >>> obs_na
    NamedArray(state=NamedArray(position=[4 5],speed=[1 2 3]),vision=[[7]
     [8]
     [9]])
    >>> obs_na.state
    NamedArray(position=[4 5],speed=[1 2 3])

    :param values: Nested key-value object of data. value should of type None, Numpy Array, or Torch Tensor.
    :type values: Dict
    :return: NamedArray with the same data structure as input. If values is empty, return None.
    :rtype: NamedArray
    """
    if values is None or len(values) == 0:
        return None
    for k, v in values.items():
        if isinstance(v, dict):
            values[k] = from_dict(v)
    return NamedArray(**values)


def array_like(x, value=0):
    if isinstance(x, NamedArray):
        return NamedArray(**{k: array_like(v, value) for k, v in x.items()})
    else:
        if isinstance(x, np.ndarray):
            data = np.zeros_like(x)
        else:
            assert isinstance(x, torch.Tensor), (
                "Currently, namedarray only supports"
                f" torch.Tensor and numpy.array (input is {type(x)})"
            )
            data = torch.zeros_like(x)
        if value != 0:
            data[:] = value
        return data


def __array_filter_none(xs):
    is_not_nones = [x is not None for x in xs]
    if all(is_not_nones) or all(x is None for x in xs):
        return
    else:
        example_x = xs[is_not_nones.index(True)]
        for i, x in enumerate(xs):
            xs[i] = array_like(example_x) if x is None else x


def recursive_aggregate(xs, aggregate_fn):
    """Recursively aggregate a list of namedarray instances.
    Typically recursively stacking or concatenating.

    :param xs: A list of NamedArrays or appropriate aggregation targets (e.g. numpy.ndarray).
    :type xs: List[NamedArray or Any]
    :param aggregate_fn: The aggregation function to be applied.
    :type aggregate_fn: Callable
    :return: The aggregated result with the same data type of elements in xs.
    :rtype: NamedArray or Any
    """
    __array_filter_none(xs)
    if isinstance(xs[0], NamedArray):
        entries = dict()
        for k in xs[0].keys():
            try:
                entries[k] = recursive_aggregate(
                    [x[k] for x in xs], aggregate_fn
                )
            except Exception as e:
                err_msg = (
                    f"`recursive_aggregate` fails at an entry named `{k}`."
                )
                if not all([type(x[k]) == type(xs[0][k]) for x in xs]):
                    err_msg += f" Types of elements are not the same: {[type(x[k]) for x in xs]}."
                else:
                    if isinstance(xs[0][k], NamedArray):
                        err_msg += " Elements are all `NamedArray`s. Backtrace to the above level."
                if not any([isinstance(x[k], NamedArray) for x in xs]):
                    specs = []
                    for x in xs:
                        specs.append(f"({x[k].dtype}, {tuple(x[k].shape)})")
                    err_msg += f" Specs of elements to be aggregated are: [{', '.join(specs)}]."
                raise RuntimeError(err_msg) from e
        return NamedArray(**entries)
    elif xs[0] is None:
        return None
    else:
        return aggregate_fn(xs)


def recursive_apply(x, fn):
    """Recursively apply a function to an NamedArray x.

    :param x: The instance of a namedarray subclass or an appropriate target to apply fn.
    :type x: NamedArray or Any
    :param fn: The function to be applied.
    :type fn: Callable
    :return: The result of applying fn to x, preserving metadata.
    :rtype: NamedArray or Any
    """
    if isinstance(x, NamedArray):
        entries = dict()
        for k, v in x.items():
            try:
                entries[k] = recursive_apply(v, fn)
            except Exception as e:
                err_msg = f"`recursive_apply` fails at an entry named `{k}`"
                if isinstance(v, NamedArray):
                    err_msg += ", which is a `NamedArray`. Backtrace to the above level."
                else:
                    err_msg += f" ({v.dtype}, {tuple(v.shape)})."
                raise RuntimeError(err_msg) from e
        res = NamedArray(**entries)
        res.register_metadata(**x.metadata)
        return res
    elif x is None:
        return None
    else:
        return fn(x)


def flatten(x: NamedArray) -> List[Tuple]:
    """Flatten a NamedArray object to a list containing structured names and values."""
    flattened_entries = []
    for k, v in x.items():
        if isinstance(v, NamedArray):
            flattened_entries += [(f"{k}." + k_, v_) for k_, v_ in flatten(v)]
        else:
            flattened_entries.append((k, v))
    return flattened_entries


def from_flattened(flattened_entries):
    """Construct a NamedArray from flattened names and values."""
    keys, values = zip(*flattened_entries)
    entries = dict()
    idx = 0
    while idx < len(keys):
        k, v = keys[idx], values[idx]
        if "." not in k:
            entries[k] = v
            idx += 1
        else:
            prefix = k.split(".")[0]
            span_end = idx + 1
            while span_end < len(keys) and keys[span_end].startswith(
                f"{prefix}."
            ):
                span_end += 1
            subentries = [
                (keys[j][len(prefix) + 1 :], values[j])
                for j in range(idx, span_end)
            ]
            entries[prefix] = from_flattened(subentries)
            idx = span_end
    return from_dict(entries)


def merge(xs: List[NamedArray]):
    """merge named arrays with different field into one, identical fields
    will be overwritten by the last one in the list
    """
    mixed_dict = {}
    for x in xs:
        mixed_dict.update(x.to_dict())
    return from_dict(mixed_dict)


def split(x: NamedArray, n):
    """split a named array into n parts along dim 0"""
    sz = x.length(0) // n
    return [x[sz * i : sz * (i + 1)] for i in range(n - 1)] + [
        x[sz * (n - 1) :]
    ]
