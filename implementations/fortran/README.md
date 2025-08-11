## Fortran Implementation

This reference version mirrors the features available in other languages:
- Adjacency-list `Graph` structure
- Specialized `DQueue` for bounded batch pulls
- Recursive `bmssp` solver driving a single-source shortest-path example

Build and run the example:
```bash
gfortran bmssp.f90 -o bmssp
./bmssp
```
