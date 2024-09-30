<!-- vscode-markdown-toc -->

- [Tensor Operations](#tensor-operations)
  - [Access functions](#access-functions)
    - [Primitives](#primitives)
      - [\*slice](#slice)
      - [\*gather (permute elements)](#gather-permute-elements)
    - [Extended APIs](#extended-apis)
      - [window_nd](#window_nd)
      - [slices](#slices)
  - [Permutation function](#permutation-function)
    - [\*transpose (permute axes)](#transpose-permute-axes)
  - [Meta info manipulation](#meta-info-manipulation)
    - [reshape](#reshape)
    - [squeeze](#squeeze)
    - [expand_dims](#expand_dims)
    - [split](#split)
  - [Memory functions](#memory-functions)
    - [\*copy](#copy)
    - [repeat](#repeat)
    - [concatenate](#concatenate)

<!-- vscode-markdown-toc-config
  numbering=true
  autoSave=true
  /vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

# Tensor Operations

Primitive tensor operations are functions whose arguments are all data types with no side-effect at all.

`*` before an operation means it is a performance-critical primitive that requires backend's first-class implementation and will not be looked into in XXX's program analysis. All the other operations (1) can be built out of these performance-critical primitives with at most a constant overhead; (2) manipulate meta information of data with low runtime overhead.

## Access functions

### Primitives

#### \*slice

```python
slice(x: Tensor, start: int, end: int, stride: int, axis: int) -> Tensor
```

#### \*gather (permute elements)

```python
gather(x: Tensor, indices: Tuple[int]) -> Tensor
```

### Extended APIs

Extended access APIs are wrappers of accessing primitives. It is not necessary to enumerate and implement them all. They are implemented through access primitives and are all unified into and analyzed as some form of access functions in the IR program.

#### window_nd

```python
window_nd(x: Tensor, window_size: Tuple[int],
          stride: Tuple[int], axis: int, pad_value: Union[T, None]) -> Tensor
```

_TODO(ying): pad?_

#### slices

```python
slices(x: Tensor, axis: int) -> FractalTensor[Tensor]
```

## Permutation function

### \*transpose (permute axes)

```python
transpose(x: Tensor, axes: Tuple[int]) -> Tensor
```

## Meta info manipulation

### reshape

```python
reshape(x: Tensor, new_shape: Tuple[int]) -> Tensor
```

### squeeze

```python
squeeze(x: Tensor, axis: int) -> Tensor
```

### expand_dims

```python
expand_dims(x: Tensor, axis: Tuple[int]) -> Tensor
```

### split

```python
split(x: Tensor, n: int, axis: int) -> Tuple[Tensor]
```

## Memory functions

### \*copy

```python
copy(x: Tensor) -> Tensor
```

### repeat

```python
repeat(x: Tensor, n: int, axis: int)-> Tensor
```

### concatenate

```python
concatenate(xs: Tuple[Tensor], axis: int) -> Tensor
```
