<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Parallel functions on FractalTensor: A demonstrating example](#parallel-functions-on-fractaltensor-a-demonstrating-example)
  - [What is Grid RNN](#what-is-grid-rnn)
  - [Why parallel functions](#why-parallel-functions)
  - [Stacked Grid RNN module by module with parallel functions](#stacked-grid-rnn-module-by-module-with-parallel-functions)
    - [1. grid cell](#1-grid-cell)
    - [nested scans](#nested-scans)
      - [2. scan along the y-direction](#2-scan-along-the-y-direction)
      - [3. scan along the x-direction](#3-scan-along-the-x-direction)
      - [4. fold to form the depth](#4-fold-to-form-the-depth)
    - [5. map the batched input](#5-map-the-batched-input)
    - [Put things together](#put-things-together)
  - [Summary of parallel functions](#summary-of-parallel-functions)
  - [Elements of the frontend program](#elements-of-the-frontend-program)
    - [Types](#types)
    - [Operations on types](#operations-on-types)
      - [Jagged FractalTensor and FractalTensor operations](#jagged-fractaltensor-and-fractaltensor-operations)
      - [Tensor and Tensor operations](#tensor-and-tensor-operations)
- [Reference](#reference)

<!-- /TOC -->

# Parallel functions on FractalTensor: A demonstrating example

## What is Grid RNN

Let's construct a complicated example step by step to demonstrate the ideas in the design of parallel functions.

The connection pattern of neural networks generally falls into two kinds: feedforward and feedback connection. Recurrent neural networks (RNNs) are in fact a very broad set of models equipped with feedback connections for sequence processing. RNN models can simply be understood as a processing unit, called a cell, that is iteratively applied to continuously supplied input data from a token stream.

To design a new RNN model for a specific task, usually, three factors are considered by algorithm researchers: (1) invent cells to capture how local input is combined with history; (2) invent new connection patterns to form a sophisticated history; and finally, (3) multiple RNN layers are further stacked to introduce more non-linearity into history.

A grid RNN cell simultaneously receives inputs from and computes results for multiple directions.
It can be used in the machine translation task which translates a source language sequence into a target language sequence and the two sequences have different lengths. We take this application as a demonstrating example in this document. Interactions among words from source and target language sequences are learned and captured by a grid RNN layer. This shares a similar idea with the extremely successful attention mechanism.

## Why parallel functions

Neural network models naturally exhibit a signal flow structure which is the fundamental constraint for auto-parallelization. Fig 1 below illustrates the signal flow structure of the stacked grid RNN model. The cell function (a circle in Fig 1) describes the local computational process where a minimal data unit is consumed by a machine learning model. In this example, a minimal data unit is a word from a sentence.

<p align="center">
<img src="images/grid_rnn_example/grid_rnn.png" width=50%><br>
Fig 1. Process generated by iteratively applying 2-d grid RNN cell function to input data.
</p>

It's worth noting that dataflow dependencies in Fig 1 are the only data dependencies (flow dependency or true dependency) that should be considered to schedule a neural network computation so that the computational process could be efficiently executed on a parallel computer, but most often, according to the way how a user writes the program, there are usually more data dependencies among program variables, leading to a conservative parallelization.

Functional style programming FractalTensor chooses largely relieves this pressure by getting rid of the side-effecting assignment and the program itself naturally preserves a clean dataflow structure. Nevertheless, to make program analysis more tractable, this is not enough. Iteratively apply the cell function to words from sentences produces a process whose shape is hyper rectangular. In the compile-time analysis, we would like to be able to make statements about the overall behavior of this process. This is very difficult in general, but parallel functions try to capture some typical patterns in neural network computations which are unique opportunities for optimizations.

## Stacked Grid RNN module by module with parallel functions

There are four dimensions in the entire computational process of the stacked grid rnn model: (1) the data parallelism dimension applies stacked grid rnn model to multiple sentence pairs; (2) the depth dimension stacks multiple grid rnn layers; (3) the x-direction scans along the source language sequence; (4) the y-direction scans along the target language sequence. (To visualize, Fig 1 omits the data parallelism dimension.)

Dependence vectors in neural network computation are all lexicographically positive, therefore these four dimensions form a fully permutable loop nest and can be computed in an arbitrary order[[1](#reference)]. But let's think from "local" to "global" module by module to define computations for (1) cell computation, (2) the y-direction, (3) the x-direction, (4) the depth, and (5) computations for a batch of sentence pairs.

### 1. grid cell

**cell**

In this document, we use the vanilla RNN cell for the example, but this is not fixed. A user can design any reasonable computations to be the cell processing unit.

Formula of the vanilla RNN cell is as follows:

$$\mathbf{h}_t = \text{cell}(\mathbf{h}_{t-1}, \mathbf{x}_t) = \text{tanh}(\mathbf{x}_t\mathbf{W} + \mathbf{h}_{t-1}\mathbf{U}+\mathbf{b})$$

<p align="center">
<img src="images/grid_rnn_example/cell.png" width=25%><br>
Fig 2. expression tree of the cell.
</p>

```python
def vanilla_cell(state: Tensor, cur: Tensor,
                 rnn_params: Tuple[Tensor]) -> Tensor:
    i2h, h2h, bias = rnn_params  # unpack tuple elements
    return ops.tanh(cur @ i2h + state @ h2h + bias)
```

**2-d grid cell**

$$\mathbf{h} = [\mathbf{h}_{t-1}^x ; \mathbf{h}_{t-1}^y]$$

$$\mathbf{h}_t^x=\text{cell}(\mathbf{h}, \mathbf{x}_t)$$

$$\mathbf{h}_t^y=\text{cell}(\mathbf{h}, \mathbf{y}_t)$$

$[; ]$ stands for "concatenation".

<p align="center">
<img src="images/grid_rnn_example/grid_cell.png" width=30%><br>
Fig 3. expression tree of the grid cell.
</p>

Below codes implement the core computation of a 2-d grid cell.

```python
# core computation
s = ops.cat(state_x, state_y, axis=0)
h_x = vanilla_cell(x_t, s, rnn_param_x)
h_y = vanilla_cell(y_t, s, rnn_param_y)
```

### nested scans

`scan` iterates over a linearly-ordered collection to aggregate the returned result from the last execution instance with an element from the collection (_#TODO(ying): There is a problem, the returned result of `zip` is able to iterate over, but it is not a homogenous collection. Unify the concepts._). The user-function should obey `scan` 's calling convention:

1. the first argument is to communicate with the last execution instance.
   - the caller (`scan`) passes the evaluation results of the last execution instance without any change to the current execution instance through the first argument.
   - returned value of the user function should have the same type and organizational structure as its first argument.
2. the second argument is input to the current execution, an indexed element of a linearly-order collection.
3. all the other arguments are passed by the caller (`scan`) as keyword arguments. They are name alias of some variables defined and initialized outside the user function.

#### 2. scan along the y-direction

To scan along the y-direction, the grid cell computation above is required to be encapsulated into a user function which acts as a binary operator:

```python
# The user-function passed to scan acts as a binary operator.
# The first argument `state` is to communicate with
# the previous execution instance. `state` carries
# a data dependence with a distance of 1.
def grid_cell(state: Tuple[Tensor], cur_input: Tuple[Tensor, Tuple[Tensor]],
              block_params: Tuple[Tuple[Tensor]]) -> Tuple[Tensor]:
    # previous execution instance computes `h_y` and `h_x`, but only `h_y`
    # from previous evalution is consumed in the current execution instance.
    _, state_y = state
    # get state_x and inputs to direction x and y.
    state_x, (x_t, y_t) = cur_input

    # unpack tuple elements, get learnable parameters
    rnn_param_x, rnn_param_y = block_params

    # the core computation
    s = ops.cat(state_x, state_y, axis=0)
    h_x = vanilla_cell(x_t, s, rnn_param_x)
    h_y = vanilla_cell(y_t, s, rnn_param_y)

    # returned value has the same organizational structure as `state`,
    # and they will be directly passed to the next execution instance.
    return h_x, h_y  # Tuple[Tensor]
```

Now, we pass `grid_cell` to `scan` which (1) guarantees the execution order of all execution instances, (2) prepares parameters before running each execution instance, and (3) stacks results of all execution instances into `FractalTensor` (s).

<p align="center">
<img src="images/grid_rnn_example/scan_y.png" width=70%><br>
Fig 4. the signal flow structure for "scanning along the y-direction".
</p>

```python
def direction_y(
        state: Tuple[FractalTensor[Tensor]],
        cur_input: Tuple[FractalTensor[Tensor]],
        block_params: Tuple[Tuple[Tensor]]) -> Tuple[FractalTensor[Tensor]]:
    state_xs, _ = state

    zero = ops.zeros(1, hidden_dim)
    return ops.scan(
        grid_cell,
        ops.zip(state_xs, ops.zip(*cur_input)),
        initializer=(zero, zero),
        block_params=block_params)

```

`state_xs` has the type of `FractalTensor` , `cur_input` has a type of `Tuple[FractalTensor]` , inputs to the x-direction and the y-direction respectively. `state_xs` and `cur_input` are prepared and passed by the caller.

#### 3. scan along the x-direction

To scan along the x-direction, it is necessary to encapsulate the computation of scanning along the y-direction into a user function.

<p align="center">
<img src="images/grid_rnn_example/scan_x.png" width=60%><br>
Fig 5. the signal flow structure for "scanning along the x-direction".
</p>

```python
def direction_x(state: Tuple[Tuple[FractalTensor[Tensor]]],
                block_params: Tuple[Tuple[Tensor]]
                ) -> Tuple[Tuple[FractalTensor[Tensor]]]:
    # len(state[0][0]) is the length of source language sequence
    zeros: FractalTensor = ops.repeat(ops.zeros(1, hidden_dim), len(state[0][0]))
    return ops.zip(*ops.scan(
        direction_y,
        state,
        initializer=(zeros, zeros),
        block_params=block_params))
```

#### 4. fold to form the depth

```python
def stacked_grid_rnns(
        sent_pair: Tuple[FractalTensor[int]]
) -> Tuple[FractalTensor[FractalTensor[Tensor]]]:
    srcs, trgs = sent_pair

    src_encs = ops.map(
        lambda word: ops.index(ops.slices(src_embedding, axis=0), word), srcs)
    trg_encs = ops.map(
        lambda word: ops.index(ops.slices(trg_embedding, axis=0), word), trgs)

    return ops.zip(*ops.fold(
        direction_x,
        rnn_params,
        initializer=ops.zip(*ops.product(src_encs, trg_encs))))

```

### 5. map the batched input

```python
src_batch: FractalTensor[FractalTensor[int]] = dataset(batch_size)
trg_batch: FractalTensor[FractalTensor[int]] = dataset(batch_size)

# data parallelism in a mini-batch
grid_out_x, grid_out_y = ops.map(stacked_grid_rnns,
                                 ops.zip(src_batch, trg_batch))

# grid_out_x: FractalTensor[FractalTensor[FractalTensor[Tensor]]]
# grid_out_y: FractalTensor[FractalTensor[FractalTensor[Tensor]]]
```

### Put things together

Fig 6 shows the overall code structure of the stacked grid RNN model we build and the memory layout of the first returned value of the outermost `map` (the second returned value of `map` has the same layout.).

<p align="center">
<img src="images/grid_rnn_example/code_structure_and_memory.png"><br>
Fig 6. the overall code structure of stacked grid RNN and the visualization of xssss's memory layout.
</p>

The user program exhibits a clear pattern of function composition and nesting and becomes very concise with the help of nestable collection type `FractalTensor` and parallel functions. However, a straightforward materialization of each function evaluation leads to fine-grained data access and movements, and non-optimal parallelisms. Large runtime overhead makes the program far from performance.

We would like to reason about the overall runtime behaviors of the computational process produced by this kind of program, and glue these function compositions and nesting into an efficient evaluation plan.

## Summary of parallel functions

1. Parallel functions are building blocks to design parallel algorithms. In machine learning tasks, data are usually organized into some high dimensional representation form. Iterating over dimensions makes nested parallelisms prevalent in machine learning computations.
2. To allow multiple levels of parallelism, the nestable collection type jagged `FractalTensor` is required to work together with parallel functions. In FractalTensor, the great expressiveness comes from:
   1. parallel functions and `FractalTensor` is data-dependent, thus there is no need to manually pad irregular data.
   2. parallel functions and `FractalTensor` can be nested for an arbitrary depth. Nested jagged `FractalTensor` can express structural information.
3. Parallel functions are optimized loops in the backend which are designed to be high-performance. Additionally, they also serve as interfaces to restrict the way a user thinks of neural network computations. The compile-time analysis then maps the entire computational process to underlying parallel computers.

## Elements of the frontend program

### Types

1. [Primitive types](primitive_types.md)
1. [User-defined types built out of primitive types](user_defined_types.md)

### Operations on types

#### Jagged FractalTensor and FractalTensor operations

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

#### [Tensor and Tensor operations](tensor_operations.md)

# Reference

1. Wolf, Michael E., and Monica S. Lam. "[A loop transformation theory and an algorithm to maximize parallelism](https://homes.luddy.indiana.edu/achauhan/Teaching/B629/2006-Fall/CourseMaterial/1991-tpds-wolf-unimodular.pdf)." IEEE Computer Architecture Letters 2.04 (1991): 452-471.
