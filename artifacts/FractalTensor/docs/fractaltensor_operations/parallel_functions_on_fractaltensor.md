<!-- vscode-markdown-toc -->

- [Builder (ctor)](#builder-ctor)
- [Parallel functions on FractalTensor](#parallel-functions-on-FractalTensor)
  - [Apply to each](#apply-to-each)
    - [map](#map)
    - [forall](#forall)
    - [filter](#filter)
    - [filterall](#filterall)
  - [Aggregate](#aggregate)
    - [reduce](#reduce)
    - [scanl/scanr](#scanlscanr)
    - [foldl/foldr](#foldlfoldr)
  - [Useful algebric identities](#useful-algebric-identities)
  - [More about the user functions passed to parallel functions](#more-about-the-user-functions-passed-to-parallel-functions)
    - [Non-unary user function](#non-unary-user-function)
    - [User function with multiple returned values](#user-function-with-multiple-returned-values)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

# Builder (ctor)

_#TODO(ying): to add._

# Parallel functions on FractalTensor

Conventions

In this document, without a specification:

1. Since`FractalTensor` can be nested, parallel and primitive operations are applied to elements at **the outermost depth** of a `FractalTensor` by default.
2. The type parameter `T` stands for all valid elementary types a `FractalTensor` can hold, that is `T = Union[Tensor, FractalTensor, int]`.

Parallel functions listed in this section require parallel implementations in the backend. Putting numerical value precision aside, parallel execution has exactly the same result as their sequential implementations, so a user could debug in sequential mode and execute in concurrent for performance. And, most importantly, the composition of parallel functions is data parallelism.

## Apply to each

Apply-to-each operations include map, forall and filter. They are a perfect fit for massively parallel machines.

For all apply-to-each operations, there is no way to communicate among execution instances, which indicates the underlying implementation of `apply-to-each` operations can execute the execution instances in any order it chooses.

### map

Type and definition of the `map` in list comprehension are:

$$\mathbf{map} ::(\alpha \rightarrow \beta) \rightarrow \Psi n.[\alpha]^d_n \rightarrow \Psi n.[\beta]^d_n$$
$$\mathbf{map} \ f \ \textit{xs} = \left[f \ x \ \big| \ x \leftarrow \textit{xs} \right]$$

Apply a unary function `f` to each element at **the outermost depth** of a (nested) `FractalTensor` .

```python
map(f:Callable[[T1], T2],
    itr: FractalTensor[T1], *args, **kwargs) -> FractalTensor[T2]
```

Example:

```python
map(f, itr=[[X, X, X, X], [X, X]])
```

is equivilent to:

```python
[f([X, X, X, X]), f([X, X])]
```

### forall

Type and definition of `forall` in list comprehension are:

$$\mathbf{forall} ::(\alpha \rightarrow \beta) \rightarrow \Psi n.[\alpha]_{n}^{d} \rightarrow \Psi n.[\beta]_{n}^{d}$$
$$\mathbf{forall} \ f \ \textit{xs} = \left[f \ x^0 \ \big| x^0 \leftarrow \textit{xs} \right] \text{  TODO: refine this notation}$$

```python
forall(f:Callable[[T1], T2], itr: FractalTensor[T1], *args, **kwargs) -> FractalTensor[T2]
```

Apply a function `f` to elements at **all depths** of a FractalTensor `TA` .

```python
forall(f, itr=[[X, X, X, X], [X, X]])
```

is equivilent to:

```python
[[f(X), f(X), f(X), f(X)], [f(X), f(X)]]
```

### filter

The first argument of `filter` is a user function that returns a bool, called the predicate denoted as $p$. `filter` returns a FractalTensor constructed from applying a predicate to each element at **the outermost depth** of the second argument, a FractalTensor `itr`. The returned FractalTensor holds elements in `itr` that fulfill the condition given by `predicate`.

Type of `filter` and its definition is:

$$\textbf{filter} ::(\alpha \rightarrow \text{bool}) \rightarrow \Psi n.[\alpha]_n^d \rightarrow \Psi m.[\alpha]_m^d$$
$$\textbf{filter} \ p \ \textit{xs} = \left[ x \ \big|\ x \leftarrow \textit{xs};\ p \ x \right]$$

```python
filter(predicate: Callable[[T], bool],
       itr: FractalTensor[T], *args, **kwargs) -> FractalTensor[T]
```

### filterall

Returns a `FractalTensor` constructed from elements of `itr` (the second argument, a `FractalTensor` ) that fulfill a condition given by `predicate` applied to each element at **all depths** of `itr` .

Type of `filterall` is:

$$\textbf{filterall} ::(\alpha \rightarrow \text{bool}) \rightarrow \Psi n.[\alpha]_n^d \rightarrow \Psi m.[\alpha]_m^d$$
$$\textbf{filterall} \ p \ \textit{xs} = \left[ x \ \big| \ x^0 \leftarrow \textit{xs};\ p \ x^0 \right] \text{  TODO: refine this notation}$$

```python
filterall(predicate: Callable[[T], bool],
          itr: FractalTensor[T], *args, **kwargs) -> FractalTensor[T]
```

## Aggregate

### reduce

The execution order of `reduce` instances is guaranteed by the order of `FractalTensor` elements.

Type and definition of `reduce` are:

$$\textbf{reduce} ::(\alpha, \alpha \rightarrow \alpha) \rightarrow \alpha \rightarrow \Psi n.[\alpha]_n^d \rightarrow [\alpha]_1^{d-1}$$
$$\mathbf{reduce} \oplus I \ \textit{xs} = I \oplus x_0 \oplus x_1  \dotsb \oplus x_n $$

where $\oplus$ is a **binary associative** user function has a type of $(\alpha,\alpha)\rightarrow \alpha$ and $I$ is the initializer. Compared with `scan` and `fold` below, the disposition of brackets in `reduce` does not affect the correctness of the evaluation result. 

```python
reduce(f:Callable[[T, T], T],
       itr: FractalTensor[T],
       initializer:T, *args, **kwargs) -> T
```

### scanl/scanr

Type and definition of scan are:

$$ \textbf{scanr} ::(\alpha, \beta \rightarrow \alpha) \rightarrow \alpha \rightarrow \Psi n.[\beta]^d_n \rightarrow \Psi n.[\alpha]_n^d$$

$$
\mathbf{scanr} \ \oplus \ I \ \textit{xs} = \left[
  (x_0 \oplus (x_{n-2} \oplus (x_{n-1} \oplus I)))
  ,\ \dotsb, \ (x_{n-2} \oplus (x_{n-1} \oplus I)),
  \ x_{n-1} \oplus I
\right]
$$


$$ \textbf{scanl} ::(\alpha, \beta \rightarrow \beta) \rightarrow \alpha \rightarrow \Psi n.[\alpha]^d_n \rightarrow \Psi n.[\beta]_n^d$$
$$

\mathbf{scanl} \ \oplus \ I \ \textit{xs} = \left[
  I \oplus x_0,\  ((I \oplus x_0) \oplus x_1), \ \dotsb,
  \ (((I \oplus x_0)\oplus x_1)\dotsb \oplus x_{n-1})
\right]
$$

where $\oplus$ is a binary associative user function has a type of $(\alpha,\beta)\rightarrow \beta$ and $I$ is the initializer. If $\oplus$ is left associative, scanl is used. If $\oplus$ is right associative, scanr is used.

```python
# T1 is not necessarily to be type equivalent to T2.
scanl(f:Callable[[T1, T2], T1],
      itr: FractalTensor[T2],
      initializer:T1, *args, **kwargs) -> FractalTensor[T1]

scanr(f:Callable[[T1, T2], T1],
      itr: FractalTensor[T2],
      initializer:T1, *args, **kwargs) -> FractalTensor[T1]
```

### foldl/foldr

$$ \textbf{foldr} ::(\alpha, \beta \rightarrow \alpha) \rightarrow \alpha \rightarrow \Psi n.[\beta]^d_n \rightarrow [\alpha]^{d-1}_1$$

$$\mathbf{foldr} \oplus I \ \textit{xs} = (x_0 \oplus ... (x_{n-1}\oplus (x_n \oplus I)) $$

$$ \textbf{foldl} ::(\alpha, \beta \rightarrow \beta) \rightarrow \alpha \rightarrow \Psi n.[\alpha]^d_n \rightarrow [\beta]^{d-1}_1$$

$$\mathbf{foldl} \oplus I \ \textit{xs} = (((I \oplus x_0) \oplus x_1) ... \oplus x_n) $$

`fold(x,...)` is equivalent to: `tail(scan(x, ...))` . It returns the result of the last `scan` instance.

- If `fold` is used when gradient computation is required, it behaves exactly the same as `scan` but only returns the result of the last execution instance.
- If `fold` is used when gradient computation is not required, since it is only necessary to return the last evaluation of the `scan` instance, execution instances could reuse memory.

```python
# T1 is not necessarily to be type equivalent to T2.
foldl(f:Callable[[T1, T2], T1],
      itr: FractalTensor[T2],
      initializer:T1, *args, **kwargs) -> T1

foldr(f:Callable[[T1, T2], T1],
      itr: FractalTensor[T2],
      initializer:T1, *args, **kwargs) -> T1
```

_**Differences between `reduce` and `fold`**_

1. `fold` is a generalized version of `reduce` in the sense that: `reduce` takes a binary associative operation $\alpha \rightarrow \alpha \rightarrow \alpha$ while `fold` takes a binary operator that is only left or right associative $\alpha \rightarrow \beta \rightarrow \alpha$. `fold` does not require two arguments of the binary operator to have the same type.
    - If `fold` is used, the analyzer think that $x \oplus (y \oplus z) = (x \oplus y) \oplus z$ is not satisfy.
1. `reduce` could be executed with more parallelism than `fold` since the latter imposes more requirements on the execution order.

*_`fold` and `scan` both have a variant that does not take the initialzier $I$._

## Useful algebric identities

_TODO(ying): to summarize._

## More about the user functions passed to parallel functions

### Non-unary user function

All parallel functions in the apply-to-each group take only unary functions. For non-unary case, `zip` **SHOULD** be used to pack multiple arguments into a single operand (_TODO(ying), what is the type of `zip` 's returned value?_).

```python
def f(*inputs) -> T:
  x, y, z = inputs
  ....
  return rv

rvs = ops.map(f, ops.zip(xs, ys, zs))
```

### User function with multiple returned values

Commonly, a user function may return multiple results. Below is the grid rnn cell example passed to `scan` .

```python
def grid_cell(...) -> Tuple[Tensor, Tuple[Tensor]]:
    ....
    s = ops.cat(state_x, state_y, axis=0)
    h_x = vanilla_cell(x_t, s, rnn_param_x)
    h_y = vanilla_cell(y_t, s, rnn_param_y)
    return h_x, h_y

# h_xs: FractalTensor[Tensor], h_ys: FractalTensor[Tensor]
h_xs, h_ys = ops.scan(grid_cell, ...)
```

Each execution instance of `grid_cell` produces two tensors: `h_x` and `h_y` . After `scan` is completed, `h_x` and `h_y` returned by all execution instances are stacked into `FractalTensor` variables: `h_xs` and `h_ys` .
