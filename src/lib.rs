use std::{sync::Arc, fmt::Display, rc::Rc, cell::RefCell};

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[derive(Clone)]
pub struct Edge<T: Display> {
    /// The "cost" of moving along this edge
    pub weight: i32,
    /// The parent of this edge
    pub parent: Rc<RefCell<Node<T>>>,
    /// Where this edge lands
    pub target: Rc<RefCell<Node<T>>>,
}

impl<T: Display> Edge<T> {
    fn new(weight: i32, parent: Rc<RefCell<Node<T>>>, target: Rc<RefCell<Node<T>>>) -> Self {
        Self {
            weight,
            parent,
            target,
        }
    }
}

#[derive(Clone)]
pub struct Node<T: Display> {
    id: T,
    edges: Vec<Edge<T>>,
}

impl<T: Display> Display for Node<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id)
    }
}

impl<T: Display> Node<T> {
    fn new(id: T) -> Self {
        Self {
            id,
            edges: Vec::new(),
        }
    }

    /// Adds an edge to the node<
    fn add_edge(&mut self, edge: Edge<T>) {
        self.edges.push(edge)
    }
}

pub struct Graph<T: Display> {
    graphs: Vec<Vec<Rc<RefCell<Node<T>>>>>,
}

impl<'a, T: Display + Clone> Graph<T> {
    fn new() -> Self {
        Self {
            graphs: Vec::new(),
        }
    }

    fn add_node(&'a mut self, x: usize, y: usize, node: Node<T>) {
        if self.graphs.get(y).is_none() {
            self.graphs.push(Vec::new());
        }
        self.graphs[y][x] = Rc::new(RefCell::new(node))
    }

    fn add_edge(&mut self, parent_coordinates: (usize, usize), target_coordinates: (usize, usize)) {
        let parent = Rc::new(RefCell::clone(&self.graphs.get(parent_coordinates.1).unwrap().get(parent_coordinates.0).unwrap()));
        let target = Rc::new(RefCell::clone(&self.graphs.get(target_coordinates.1).unwrap().get(target_coordinates.0).unwrap()));
        self.graphs[parent_coordinates.1][parent_coordinates.0].borrow_mut().add_edge(Edge::new(1, parent, target));
    }
}

impl<T: Display> Display for Graph<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut graph = String::new();
        for y in &self.graphs {
            for x in y {
                for edge in &x.borrow().edges {
                    graph.push_str(&format!("{} -> {}", edge.parent.borrow().id, edge.target.borrow().id));
                }
                graph.push('\n');
            }
        }
        write!(f, "{}", graph)
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

    #[test]
    fn some_test() {
        let mut graph = Graph::new();
        graph.add_node(0, 0, Node::new(1));
        graph.add_node(0, 1, Node::new(2));
        graph.add_edge((0,0), (0,1));
        println!("{}", graph);
    }
}
