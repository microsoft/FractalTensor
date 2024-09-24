<!-- vscode-markdown-toc -->

- [Index-related operations](#index-related-operations)
	- [arange](#arange)
- [List comprehensions for describing index sets](#list-comprehensions-for-describing-index-sets)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

# Index-related operations

## arange

```python
arange(start:int, end:int, stride:int) -> FractalTensor[int]
```

Generate index vector. This operation is denoted as:

1. $[a..b]$ which a list of integers in in­creasing order from $a$ to $b$ inclusive, going up in step of 1. If $a > b$, then $[a..b] = [\ ]$.
1. $[a,b..c]$ called arithmetic progression, $[a, a+d, a+2\times d,...,c]$ where $d=b-a$.

# List comprehensions for describing index sets

In XXX, list comprehension adapts a syntax from conventional mathematics particularly for (1) **describing mapping between index sets**; (2) **generating access patterns**. This is to make it possible for the compiler to track element permutation between two collection types in a concise mathematical representation.

The abstract syntax is:

$[< \textit{expression}> | <\textit{qualifier}>; ...; <\textit{qualifier}>]$

where:

- $<expression>$ denotes an expression that **has a quasi-affine form** which **ONLY** include operations from the below set:
  - $+::\text{int} \rightarrow \text{int} \rightarrow \text{int}$
  - scaling with a plain integer constant
  - integer div with a plain integer constatn
  - mod with a plain integer constant
- a $<qualifier>$ is a generator having a form of:
  - $<\textit{variable}::\text{int}> \leftarrow <\textit{list}>$
  - $<\textit{variable}::\text{int},...,\textit{variable}::\text{int}> \leftarrow <\textit{listoftuples}>$

Examples:

> $[i \times i \ | \ i \leftarrow [1..10]]$
>
> $[i \ | \ i \leftarrow [1..10] \textbf{ even } i]$
>
> $[i \ | \ i \leftarrow [1..n]; i \textbf{ mod } d = 0]$
>
> $[(i, j) \ |\  i \leftarrow [1..3]; j\leftarrow [1..2]]$
>
> $[(i,j) \ | \ i \leftarrow [1..4]; \textbf{ even } i; j \leftarrow [i+4]; \textbf{ odd }j]$
