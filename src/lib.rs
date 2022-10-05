use std::{fmt::Display, rc::Rc, cell::RefCell};

/// An edge between two nodes inside the graph
#[derive(Clone)]
pub struct Edge<T: Display> {
    /// The "cost" of moving along this edge
    pub weight: i32,
    /// The parent of this edge
    pub parent: Rc<RefCell<Node<T>>>,
    /// Where this edge lands
    pub target: Rc<RefCell<Node<T>>>,
}

impl<T: Display + Eq> Edge<T> {
    /// Creates a new edge
    /// # Params
    /// - `weight` the weight of this edge
    /// - `parent` the node from which the edge originates
    /// - `target` the node to which the edge lands
    pub fn new(weight: i32, parent: Rc<RefCell<Node<T>>>, target: Rc<RefCell<Node<T>>>) -> Self {
        Self {
            weight,
            parent,
            target,
        }
    }
}

/// A node inside the graph
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
    /// Creates a new node
    /// `id` - An identifier for this node, should be unique
    pub fn new(id: T) -> Self {
        Self {
            id,
            edges: Vec::new(),
        }
    }

    /// Adds an edge to the node
    pub fn add_edge(&mut self, edge: Edge<T>) {
        self.edges.push(edge)
    }
}

/// Structure to organize nodes with edges
pub struct Graph<T: Display> {
    nodes: Vec<Rc<RefCell<Node<T>>>>,
}

impl<'a, T: Display + Clone + Eq> Graph<T> {
    /// Creates a new graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
        }
    }

    /// Adds a new edge to the graph and returns the index of the node
    pub fn add_node(&'a mut self, node: Node<T>) -> usize {
        self.nodes.push(Rc::new(RefCell::new(node)));
        self.nodes.len()-1
    }

    /// Adds an edge between two nodes, single direction
    pub fn add_edge(&mut self, weight: i32, parent_index: usize, target_index: usize) {
        let parent = Rc::new(RefCell::clone(&self.nodes.get(parent_index).unwrap()));
        let target = Rc::new(RefCell::clone(&self.nodes.get(target_index).unwrap()));
        self.nodes[parent_index].borrow_mut().add_edge(Edge::new(weight, parent, target));
    }
    
    /// Adds an edge between two nodes, booth directions
    pub fn add_double_edge(&mut self, weight: i32, parent_index: usize, target_index: usize) {
        let parent = Rc::new(RefCell::clone(&self.nodes.get(parent_index).unwrap()));
        let target = Rc::new(RefCell::clone(&self.nodes.get(target_index).unwrap()));
        self.nodes[parent_index].borrow_mut().add_edge(Edge::new(weight, parent.clone(), target.clone()));
        self.nodes[target_index].borrow_mut().add_edge(Edge::new(weight, target, parent));
    }

    /// Searches for the id and returns the index if found.
    pub fn get_index_by_id(&self, id: T) -> Option<usize> {
        for (index, node) in self.nodes.iter().enumerate() {
            if node.borrow().id.eq(&id) {
                return Some(index);
            }
        }
        None
    }
}

impl<T: Display> Display for Graph<T> {
    /// Formats the graph to show all edges between nodes
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut graph = String::new();
        for node in &self.nodes {
            let id = &node.borrow().id;
            graph.push_str(&format!("{}: ", id));
            for edge in &node.borrow().edges {
                graph.push_str(&format!("(--{}-> {})", edge.weight, edge.target.borrow().id));
            }
            graph.push('\n');
        }
        write!(f, "{}", graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn some_test() {
        let mut graph = Graph::new();
        graph.add_node(Node::new("Siegburg"));
        graph.add_node(Node::new("Bonn"));
        graph.add_node(Node::new("Köln"));
        graph.add_node(Node::new("Troisdorf"));
        graph.add_edge(5, 0, 1);
        graph.add_edge(7, 0, 2);
        graph.add_double_edge(10, 1, 2);
        graph.add_edge(2, graph.get_index_by_id("Troisdorf").unwrap(), graph.get_index_by_id("Köln").unwrap());
        println!("{}", graph);
    }
}
