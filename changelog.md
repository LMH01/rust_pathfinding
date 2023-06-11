# v4.0.0-dev

- Implemented Bellman-Ford algorithm
- Reorganized internal structure
- Updated function Graph::node_by_id to take a reference instead of the actual value

# v3.0.0

- Added option to parse graph from a list of instructions
- Added feature "steps", when enabled some information is shown when dijkstra is run
- Removed debug feature

## Bugfixes

- Fixed dijkstra implementation to also compute the correct path when edges are already in the open list but with a larger distance
- Shortest path is now also reset when nodes are reset, this could cause problems