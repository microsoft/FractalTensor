<!-- vscode-markdown-toc -->

- [Access primitives of FractalTensor](#access-primitives-of-FractalTensor)
  - [\*index](#index)
  - [\*slice](#slice)
  - [\*gather (permute elements)](#gather-permute-elements)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->


# Access primitives of FractalTensor

Primitive access operations have first-class implementations in the backend.

## \*index

$$\mathbf{index} ::\Psi n.[\alpha]^d_n \rightarrow \Psi m.[\alpha]^{d-1}_m$$

```python
index(x: FractalTensor[T], i: int) -> T
```

Access a `FractalTensor` variable using the `[]` operator is equivalent to call `index` .

##  \*slice

```python
slice(x: FractalTensor[T], start: int, end: int, stride: int) -> FractalTensor[T]
```

##  \*gather (permute elements)

```python
gather(x: FractalTensor[T], indices:Tuple[int]) -> FractalTensor[T]
```
