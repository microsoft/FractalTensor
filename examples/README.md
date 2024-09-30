# Restrictions of using Python's syntax in implementing the examples

- The parser parses dataflow relations among variable generations and usages from
the source code. Functions can be nested, but the current implementation of the parser ONLY
resolves name alias in the function's local scope. Name resolver does not search
the enclosing scope. A better implementation of name resolver will remove this limitation.
