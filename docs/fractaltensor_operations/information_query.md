<!-- vscode-markdown-toc -->

- [Information query](#information-query)
  - [length](#length)
  - [depth](#depth)

<!-- vscode-markdown-toc-config

    numbering=true
    autoSave=true
    /vscode-markdown-toc-config -->

<!-- /vscode-markdown-toc -->

## Information query

### length

$$\mathbf{length} ::\Psi n.[\alpha]^d_n \rightarrow \text{int}$$

```python
length(x: FractalTensor[T]) -> List[int]
```

`length` is only available after data is feed to a `FractalTensor` variable, otherwise, return an empty list.

### depth

$$\mathbf{depth} ::\Psi n.[\alpha]^d_n \rightarrow \text{int}$$

```python
depth(x: FractalTensor[T]) -> int
```
