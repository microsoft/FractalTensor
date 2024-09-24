<!-- vscode-markdown-toc -->

- [Elements of the frontend program](#elements-of-the-frontend-program)
  - [Types](#types)
  - [Operations on types](#operations-on-types)
    - [Jagged FractalTensor and FractalTensor operations](#jagged-FractalTensor-and-FractalTensor-operations)
    - [Tensor and Tensor operations](#tensor-and-tensor-operations)
- [Internal representation](#internal-representation)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

# Elements of the frontend program

## Types

1. [Primitive types in XXX](primitive_types.md)
1. [User-defined types built out of primitive types](user_defined_types.md)
1. [A summary: abstract syntax and the type system](abstract_syntax/README.md)

## Operations on types

### Jagged FractalTensor and FractalTensor operations

1. [Memory layout of FractalTensor](fractaltensor_operations/memory_layout_of_fractaltensor.md)
1. Data parallel patterns on FractalTensor

   Parallel functions are high-order functions, the first argument of which is a user function, and the second argument of which is a `FractalTensor` or an `Iterator` (_TODO(ying)_). Parallel functions iteratively apply the user-defined function to elements of the second argument. We call each evaluation of the user-defined function upon a part of elements of the `FractalTensor` _**an instance**_.

   - [Parallel functions on FractalTensor: a demonstrating example](fractaltensor_operations/parallel_functions_example.md)
   - [Parallel functions: types and semantics](fractaltensor_operations/parallel_functions_on_fractaltensor.md)

1. [Information query](fractaltensor_operations/information_query.md)
1. Access operations of FractalTensor

   Throughout the documents, `*` before an operation means it is a performance-critical operation that requires backend's first-class implementation and will not be looked into in XXX's program analysis. All the other operations (1) can be built out of these performance-critical primitives with at most a constant overhead; or (2) manipulate meta information of data with low runtime overhead.

   - [Access primitives](fractaltensor_operations/access_primitives.md)
   - [Extended access operations](fractaltensor_operations/extended_access_operations.md)
   - [Access multiple FractalTensors simultaneously](fractaltensor_operations/access_multiple_factaltensors.md)

### [Tensor and Tensor operations](tensor_operations.md)

# [Internal representation](internal_representation/README.md)

[TBD]
