<!-- vscode-markdown-toc -->

- [User-defined types built out of primitive types](#user-defined-types-built-out-of-primitive-types)
  - [Motivations](#motivations)
  - [The name alias and indirect reference issue](#the-name-alias-and-indirect-reference-issue)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

# User-defined types built out of primitive types

## Motivations

`Tensor` and `FractalTensor` are all homogenous arrays that participated in arithmetic operations. Operations on them require high-performance on parallel computers. For programmability, due to the constraint that `Tensor` and `FractalTensor` elements should be homogenous so that the underlying memory is compact, it is easy to find that it will be cumbersome if only `Tensor` and `FractalTensor` are allowed to program.

By using a simplified example, this document first only discusses the immutable array of names. At the moment, we just think a machine learning model is a pure function $\mathbf{ys} = f (\mathbf{xs}, \mathbf{\omega})$ which evaluates values of outputs $\mathbf{ys}$ by given values of its inputs $\mathbf{xs}$ and parameters $\mathbf{\omega}$. Below is the vanilla RNN cell example, a very simple machine learning "model".

Example of the vanilla RNN cell:

```python
def vanilla_cell(x: Tensor, state: Tensor, i2h: Tensor, h2h: Tensor,
                 bias: Array) -> Tensor:
    return ops.tanh(x @ i2h + state @ h2h + bias)
```

In this simplified code snippet, `x` is the input, while `state`, `ih2`, `h2h`, and `bias` are learnable parameters. What the program would look like if the model $f$ contains dozens of learnable parameters? What the program would look like if the model $f$ is a complex function composition of dozens of more primitive functions?

In machine learning tasks, learnable parameters often require various special treatments (like updating values, applying regularizers, clipping gradients of them). It would be convenient if learnable parameters that have some logical relations, for example, learnable parameters of a neural network layer, learnable parameters of a building block, could be packed into a single variable and managed as a whole. This will make the program clean, and easy to manage and evolve. Learnable parameters are tensors with different shapes. This violates the constraint of `FractalTensor`, so they cannot be packed into a `FractalTensor`.

To solve this problem, in the code region that defines neural network computations (a code region XXX analyzes), XXX **narrowly interprets** Python's built-in `Tuple` of `Tensor` and `FractalTensor` as an immutable collection of **names**. A unique name indicates a unique value. See the below example:

Example of the vanilla RNN cell that packs learnable parameters together.

```python
def vanilla_cell(x: Tensor, state, rnn_param: Tuple[Tensor]) -> Tensor:
    i2h, h2h, bias = rnn_param  # unpack tuple elements
    return ops.tanh(input @ i2h + state @ h2h + bias)
```

Moreover, like `FractalTensor`, **the immutable array of names can be nested** so that a set of variables having different types can be logically bundled together to form a single new variable.

> _The array of names we discussed here could be elegantly implemented as a struct, named tuple, or other types in a language. In XXX, we focus on figuring out the "just right" constraints on how to interpret behaviors of the proposed types so that to strike the right balance that makes the program not only naturally expressive but also amenable to compiler optimizations for parallel and efficient execution, so we simply leverage the host language Python's built-in type Tuple (named Tuple?) to mimic the "user-defined type"._

There is one requirement: a user-defined type should support the `index` method.

## The name alias and indirect reference issue

_TODO(ying), This is a preprocessing problem that depends on some implementation choices. Refine and rethink this later._

The immutable array of names creates a name alias, which in turn, leads to an indirect reference of the same variable. To auto-parallelize and auto-differentiate of user's program, the compiler has to identify operations on the same variable regardless of their name alias in original codes. In XXX, **assignment is interpreted as parameter binding to associate value with variable**. Binding is a weaker operation than the side-effecting assignment. Analysis of binding is more tractable. Besides, the program under analysis is only allowed to use meta operations that are all pure.

Due to these constraints, a name resolver can replace the immutable array of names into unified names when the original program is lowered to the IR program. The immutable array of names is not a back-end implemented and optimized type. It is to improve programmability at the front-end.
