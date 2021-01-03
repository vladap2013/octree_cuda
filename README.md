# Octree implementation with CUDA support

Implementation of the Octree which can be queried from the CUDA device code.

The implementation is based on the [paper](http://jbehley.github.io/papers/behley2015icra.pdf): 
J. Behley, V. Steinhage, A.B. Cremers. *Efficient Radius Neighbor Search in Three-dimensional Point Clouds*, Proc. of the IEEE International Conference on Robotics and Automation (ICRA), 2015.

Ideas from CPU Octree implementation at https://github.com/jbehley/octree are used. That implementation is more optimal
and feature rich in case CUDA is not used.
