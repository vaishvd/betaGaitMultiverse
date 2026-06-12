"""
Decision nodes for the betaGaitMultiverse multiverse analysis.

Each node is a pure function: it takes data plus a decision parameter
and returns transformed data. Nodes have no file I/O and no side effects,
so they can be composed into any pipeline branch and reused by both the
canonical pipeline and the COMET multiverse.
"""
