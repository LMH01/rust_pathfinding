use std::{sync::Arc, fmt::Display};

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

pub struct Edge<'a, T: Display> {
    /// The "cost" of moving along this edge
    pub weight: i32,
    /// The parent of this edge
    pub parent: Arc<&'a Node<'a, T>>,
    /// Where this edge lands
    pub target: Arc<&'a Node<'a, T>>,
}

impl<'a, T: Display> Edge<'a, T> {
    fn new(weight: i32, parent: Arc<&Node<T>>, target: Arc<&Node<T>>) -> Self {
        Self {
            weight,
            parent,
            target,
        }
    }
}

pub struct Node<'a, T: Display> {
    id: T,
    edges: Vec<Edge<'a, T>>,
}

impl<T: Display> Display for Node<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id)
    }
}

impl<T: Display> Node<'_, T> {
    /// Adds an edge to the node<
    fn add_edge(&mut self, edge: Edge<T>) {
        self.edges.push(edge)
    }
}

pub struct Graph<'a, T: Display> {
    graphs: Vec<Vec<Node<'a, T>>>,
}

impl<'a, T: Display> Graph<'a, T> {
    fn new() -> Self {
        Self {
            graphs: Vec::new(),
        }
    }

    fn add_node(&'a mut self, x: usize, y: usize, node: Node<T>) {
        if self.graphs.get(y).is_none() {
            self.graphs.push(Vec::new());
        }
        self.graphs[y][x] = node
    }

    fn add_edge(&mut self, parent_coordinates: (usize, usize), target_coordinates: (usize, usize)) {
        let parent = Arc::new(self.graphs.get(parent_coordinates.1).unwrap().get(parent_coordinates.0).unwrap());
        let target = Arc::new(self.graphs.get(target_coordinates.1).unwrap().get(target_coordinates.0).unwrap());
        self.graphs[parent_coordinates.1][parent_coordinates.0].add_edge(Edge::new(1, parent, target));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
