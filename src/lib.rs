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

impl<T: Display + Eq> PartialEq for Edge<T> {
    fn eq(&self, other: &Self) -> bool {
        self.parent.borrow().id.eq(&other.parent.borrow().id) 
            && self.target.borrow().id.eq(&other.target.borrow().id)
            && self.weight.eq(&other.weight)
    }
}

/// A node inside the graph
#[derive(Clone)]
pub struct Node<T: Display> {
    id: T,
    edges: Vec<Edge<T>>,
    distance: i32,
    shortest_path: Vec<Rc<RefCell<Node<T>>>>,
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
            distance: i32::MAX,
            shortest_path: Vec::new(),
        }
    }

    /// Adds an edge to the node
    pub fn add_edge(&mut self, edge: Edge<T>) {
        self.edges.push(edge)
    }

    /// Returns the distance of this node.
    pub fn distance(&self) -> i32 {
        self.distance
    }

    /// Sets the total distance of this node, used when running the shortest path algorithm.
    pub fn set_distance(&mut self, distance: i32) {
        self.distance = distance;
    }
}

impl<T: Display + Eq> PartialEq for Node<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
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
        let parent = Rc::clone(&self.nodes.get(parent_index).unwrap());
        let target = Rc::clone(&self.nodes.get(target_index).unwrap());
        //let parent = Rc::new(RefCell::clone(&self.nodes.get(parent_index).unwrap()));
        //let target = Rc::new(RefCell::clone(&self.nodes.get(target_index).unwrap()));
        self.nodes[parent_index].borrow_mut().add_edge(Edge::new(weight, parent, target.clone()));
    }
    
    /// Adds an edge between two nodes, booth directions
    pub fn add_double_edge(&mut self, weight: i32, parent_index: usize, target_index: usize) {
        let parent = Rc::clone(&self.nodes.get(parent_index).unwrap());
        let target = Rc::clone(&self.nodes.get(target_index).unwrap());
        //let parent = Rc::new(RefCell::clone(&self.nodes.get(parent_index).unwrap()));
        //let target = Rc::new(RefCell::clone(&self.nodes.get(target_index).unwrap()));
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

    /// Calculates the shortest distance between two nodes.
    /// # Params
    /// `start_id` - The id of the start node
    /// `target_id` - The id of the target node
    /// # Panics
    /// Panics when the graph does not contain the start or target id.
    /// # Returns
    /// `Some(length)` when the shortest path was found.
    /// `None` when no path between the two nodes exists.
    pub fn shortest_path_djikstra(&mut self, start_id: T, target_id: T) -> Option<i32> {
        let start = &self.nodes[self.get_index_by_id(start_id).unwrap()].clone();
        let target = &self.nodes[self.get_index_by_id(target_id).unwrap()].clone();
        self.reset_nodes();
        start.borrow_mut().set_distance(0);
        let mut open_nodes: Vec<Rc<RefCell<Node<T>>>> = Vec::new();
        let mut closed_nodes: Vec<Rc<RefCell<Node<T>>>> = Vec::new();
        open_nodes.push(start.clone());

        while !open_nodes.is_empty() {
            let node = pop_lowest_distance_node(&mut open_nodes).unwrap();
            for edge in &node.borrow().edges {
                let target = &edge.target;
                let edge_weight = edge.weight;
                if !closed_nodes.contains(target) {
                    calc_min_distance(target, edge_weight, &node);
                    open_nodes.push(target.clone());
                }
            }
            closed_nodes.push(node);
        }

        let target_distance = target.borrow().distance;
        if target_distance == i32::MAX {
            None
        } else {
            Some(target_distance)
        }
    }

    /// Resets the distance of each node in the graph back to `i32::MAX`.
    /// 
    /// Should be called after `shortest_path_djikstra` has run, otherwise the next call might not result in the correct distance.
    fn reset_nodes(&mut self) {
        for node in self.nodes.iter_mut() {
            node.borrow_mut().set_distance(i32::MAX);
        }
    }
}

/// Removes the node with the lowest distance and returns it.
fn pop_lowest_distance_node<T: Display>(nodes: &mut Vec<Rc<RefCell<Node<T>>>>) -> Option<Rc<RefCell<Node<T>>>> {
    let mut lowest_distance_node: Option<Rc<RefCell<Node<T>>>> = None;
    let mut lowest_distance = i32::MAX;
    let mut index_to_remove: Option<usize> = None;
    for (index, node) in nodes.iter().enumerate() {
        let node_distance = node.borrow().distance();
        if node_distance < lowest_distance {
            lowest_distance = node_distance;
            lowest_distance_node = Some(node.clone());
            index_to_remove = Some(index);
        }
    }
    nodes.remove(index_to_remove.unwrap());
    lowest_distance_node
}

fn calc_min_distance<T: Display>(node: &Rc<RefCell<Node<T>>>, weight: i32, source: &Rc<RefCell<Node<T>>>) {
    let source_distance = source.borrow().distance();
    if source_distance + weight < node.borrow().distance() {
        node.borrow_mut().set_distance(source_distance + weight);
        let mut shortest_path = source.borrow().shortest_path.clone();
        shortest_path.push(source.clone());
        node.borrow_mut().shortest_path = shortest_path;
    }
}

impl<T: Display> Display for Graph<T> {
    /// Formats the graph to show all edges between nodes
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut graph = String::new();
        graph.push_str(&format!("{:13} | {:08} | edges\n", "id", "distance"));
        graph.push_str("--------------------------------------------------------------------\n");
        for node in &self.nodes {
            let id = &node.borrow().id;
            let distance = node.borrow().distance;
            if distance != i32::MAX {
                graph.push_str(&format!("{:13} | {:8} | ", id, distance));
            } else {
                graph.push_str(&format!("{:13} | {:8} | ", id, ""));
            }
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
        graph.add_node(Node::new("Bergheim"));
        graph.add_edge(5, 0, 1);
        graph.add_edge(6, 0, 2);
        graph.add_edge(2, 1, 0);
        graph.add_edge(9, 1, 3);
        graph.add_edge(7, 2, 0);
        graph.add_edge(2, 2, 3);
        graph.add_edge(5, 3, 2);
        graph.add_edge(1, 3, 1);
        graph.add_edge(10, 3, 4);
        graph.add_edge(4, 4, 0);
        println!("{}", graph);
        //println!("Length: {}", graph.shortest_path("Siegburg", "Troisdorf").unwrap());
        println!("Length: {}", graph.shortest_path_djikstra("Siegburg", "Troisdorf").unwrap());
        println!("Length: {}", graph.shortest_path_djikstra("Bonn", "Köln").unwrap());
        println!("{}", graph);
        println!("Length: {}", graph.shortest_path_djikstra("Siegburg", "Bergheim").unwrap_or(-1));
        println!("{}", graph);
    }
}
