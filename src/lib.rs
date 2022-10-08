use std::{fmt::Display, rc::Rc, cell::RefCell, collections::HashMap};

/// An edge between two nodes inside the graph
#[derive(Clone)]
pub struct Edge<T: Display> {
    /// The "cost" of moving along this edge
    weight: i32,
    /// The parent of this edge
    parent: Rc<RefCell<Node<T>>>,
    /// Where this edge lands
    target: Rc<RefCell<Node<T>>>,
}

impl<T: Display + Eq> Edge<T> {
    /// Creates a new edge
    /// # Params
    /// - `weight` the weight of this edge
    /// - `parent` the node from which the edge originates
    /// - `target` the node to which the edge lands
    fn new(weight: i32, parent: Rc<RefCell<Node<T>>>, target: Rc<RefCell<Node<T>>>) -> Self {
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

impl<T: Display + Eq> Node<T> {
    /// Creates a new node
    /// 
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
    fn add_edge(&mut self, edge: Edge<T>) {
        self.edges.push(edge)
    }

    /// Sets the total distance of this node, used when running the shortest path algorithm.
    fn set_distance(&mut self, distance: i32) {
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

    /// Returns a reference to the node or `None` if the id is not used.
    pub fn node_by_id(&self, id: T) -> Option<Rc<RefCell<Node<T>>>> {
        for node in self.nodes.iter() {
            if node.borrow().id.eq(&id) {
                return Some(node.clone());
            }
        }
        None
    }


    /// Resets the distance of each node in the graph back to `i32::MAX`.
    /// 
    /// Should be called after [djikstra](./fn.djikstra.html) has run, otherwise the next call might not result in the correct distance.
    pub fn reset_nodes(&mut self) {
        for node in self.nodes.iter_mut() {
            node.borrow_mut().set_distance(i32::MAX);
        }
    }

    /// Create a graph from a 2D vector containing i32.
    /// 
    /// The i32 value is the edge weight of each edge leading into that node.
    /// # Example
    /// ```
    /// use lmh01_pathfinding::{Graph, djikstra};
    /// 
    /// // Prepare example vector
    /// let mut vec: Vec<Vec<i32>> = Vec::new();
    /// let vec_inner_1 = vec![3, 4, 5];
    /// let vec_inner_2 = vec![1, 2, 3];
    /// let vec_inner_3 = vec![1, 8, 2];
    /// vec.push(vec_inner_1);
    /// vec.push(vec_inner_2);
    /// vec.push(vec_inner_3);
    /// 
    /// // Create graph from example vector
    /// let graph = Graph::<String>::from_i32_vec(&vec);
    /// 
    /// // Run djikstra's algorithm
    /// assert_eq!(8, djikstra(graph.node_by_id(String::from("[0|0]")).unwrap(), graph.node_by_id(String::from("[2|2]")).unwrap()).unwrap_or(-1));
    /// ```
    pub fn from_i32_vec(vec: &Vec<Vec<i32>>) -> Graph<String> {
        if cfg!(feature = "debug") {
            println!("Constructing graph from vector...");
            println!("Adding nodes...");
        }
        let mut graph = Graph::new();
        let mut index_by_id: HashMap<(usize, usize), usize> = HashMap::new();
        for (i_y, y) in vec.iter().enumerate() {
            for (i_x, _x) in y.iter().enumerate() {
                let index = graph.add_node(Node::new(String::from(format!("[{}|{}]", i_x, i_y))));
                index_by_id.insert((i_x, i_y), index);
            }
        }
        if cfg!(feature = "debug") {
            println!("Adding edges...");
        }
        for (i_y, y) in vec.iter().enumerate() {
            let max_x_size = y.len();
            for (i_x, x) in y.iter().enumerate() {
                for neighbor in neighbor_positions((i_x, i_y), max_x_size, vec.len()) {
                    graph.add_edge(*x, *index_by_id.get(&(neighbor.0, neighbor.1)).unwrap(), *index_by_id.get(&(i_x, i_y)).unwrap());
                }
            }
        }
        graph
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

/// Calculates the shortest distance between two nodes.
/// This will utilize the algorithm by [djikstra](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm).
/// 
/// The distance field in each node should be set to `i32:MAX` before this function is called.
/// When the nodes are organized using the [Graph](struct.Graph.html) struct the function [reset_nodes](struct.Graph.html#method.reset_nodes) may be used to reset the distance field.
/// # Params
/// `start_node` - The start node
/// 
/// `target_node` - The target node
/// # Returns
/// `Some(length)` when the shortest path was found.
/// 
/// `None` when no path between the two nodes exists.
/// 
/// # Example
/// ```rust
/// use lmh01_pathfinding::{Node, Graph, djikstra};
/// 
/// // Create new graph
/// let mut graph: Graph<char> = Graph::new();
/// 
/// // Add nodes to graph
/// let node_a_idx = graph.add_node(Node::new('a'));
/// let node_b_idx = graph.add_node(Node::new('b'));
/// let node_c_idx = graph.add_node(Node::new('c'));
/// let node_d_idx = graph.add_node(Node::new('d'));
/// let node_e_idx = graph.add_node(Node::new('e'));
/// 
/// // Add edges between nodes
/// graph.add_edge(3, node_a_idx, node_b_idx);
/// graph.add_edge(4, node_a_idx, node_c_idx);
/// graph.add_edge(5, node_b_idx, node_a_idx);
/// graph.add_edge(2, node_b_idx, node_d_idx);
/// graph.add_edge(9, node_c_idx, node_a_idx);
/// graph.add_edge(1, node_c_idx, node_d_idx);
/// graph.add_edge(3, node_d_idx, node_b_idx);
/// graph.add_edge(7, node_d_idx, node_c_idx);
/// 
/// // Run djikstra's algorithm to determine the shortest path, result contains the shortest distance.
/// let result = djikstra(graph.node_by_id('a').unwrap(), graph.node_by_id('d').unwrap()).unwrap_or(-1);
/// assert_eq!(5, result);
/// 
/// // Reset node distances before running the algorithm again
/// graph.reset_nodes();
/// 
/// // Run algorithm again
/// let result = djikstra(graph.node_by_id('b').unwrap(), graph.node_by_id('c').unwrap()).unwrap_or(-1);
/// assert_eq!(9, result);
/// 
/// // Reset nodes again
/// 
/// // Run algorithm again, returns -1 because no node exists that connects e to the rest of the graph.
/// let result = djikstra(graph.node_by_id('a').unwrap(), graph.node_by_id('e').unwrap()).unwrap_or(-1);
/// assert_eq!(-1, result);
/// 
/// ```
/// ```should_panic
/// use lmh01_pathfinding::{Node, Graph, djikstra};
/// 
/// let mut graph = Graph::new();
/// graph.add_node(Node::new('a'));
/// // Panics because the node b does not exist in the graph.
/// let result = djikstra(graph.node_by_id('a').unwrap(), graph.node_by_id('b').unwrap()).unwrap_or(-1);
/// ```
pub fn djikstra<T: Display + Clone + Eq>(start_node: Rc<RefCell<Node<T>>>, target_node: Rc<RefCell<Node<T>>>) -> Option<i32> {
    start_node.borrow_mut().set_distance(0);
    let mut open_nodes: Vec<Rc<RefCell<Node<T>>>> = Vec::new();
    let mut closed_nodes: Vec<Rc<RefCell<Node<T>>>> = Vec::new();
    open_nodes.push(start_node.clone());

    while !open_nodes.is_empty() {
        let node = pop_lowest_distance_node(&mut open_nodes).unwrap();
        if cfg!(feature = "debug") {
            if open_nodes.len() % 100 == 0 {
                println!("Nodes open / closed: {}/{}", open_nodes.len(), closed_nodes.len());
            }
        }
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

    let target_distance = target_node.borrow().distance;
    if target_distance == i32::MAX {
        None
    } else {
        Some(target_distance)
    }
}

/// Removes the node with the lowest distance and returns it.
fn pop_lowest_distance_node<T: Display + Eq>(nodes: &mut Vec<Rc<RefCell<Node<T>>>>) -> Option<Rc<RefCell<Node<T>>>> {
    let mut lowest_distance_node: Option<Rc<RefCell<Node<T>>>> = None;
    let mut lowest_distance = i32::MAX;
    let mut index_to_remove: Option<usize> = None;
    for (index, node) in nodes.iter().enumerate() {
        let node_distance = node.borrow().distance;
        if node_distance < lowest_distance {
            lowest_distance = node_distance;
            lowest_distance_node = Some(node.clone());
            index_to_remove = Some(index);
        }
    }
    nodes.remove(index_to_remove.unwrap());
    lowest_distance_node
}

fn calc_min_distance<T: Display + Eq>(node: &Rc<RefCell<Node<T>>>, weight: i32, source: &Rc<RefCell<Node<T>>>) {
    let source_distance = source.borrow().distance;
    if source_distance + weight < node.borrow().distance {
        node.borrow_mut().set_distance(source_distance + weight);
        let mut shortest_path = source.borrow().shortest_path.clone();
        shortest_path.push(source.clone());
        node.borrow_mut().shortest_path = shortest_path;
    }
}

/// Returns the neighboring positions for a position in a 2D graph.
/// 
/// # Example
/// ```
/// use lmh01_pathfinding::neighbor_positions;
/// 
/// let neighbors = neighbor_positions((2,2), 10, 10);
/// assert_eq!((1, 2), neighbors[0]);
/// assert_eq!((2, 1), neighbors[1]);
/// assert_eq!((3, 2), neighbors[2]);
/// assert_eq!((2, 3), neighbors[3]);
/// ```
pub fn neighbor_positions(pos: (usize, usize), max_x_size: usize, max_y_size: usize) -> Vec<(usize, usize)> {
    let mut positions = Vec::new();
    if pos.0 != 0 {
        positions.push((pos.0-1, pos.1));
    }
    if pos.1 != 0 {
        positions.push((pos.0, pos.1-1));
    }
    if pos.0 != max_x_size-1 {
        positions.push((pos.0+1, pos.1));
    }
    if pos.1 != max_y_size-1 {
        positions.push((pos.0, pos.1+1));
    }
    positions
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
        graph.add_double_edge(10, 3, 4);
        println!("Length: {}", djikstra(graph.node_by_id("Siegburg").unwrap(), graph.node_by_id("Troisdorf").unwrap()).unwrap());
        graph.reset_nodes();
        println!("Length: {}", djikstra(graph.node_by_id("Bonn").unwrap(), graph.node_by_id("Köln").unwrap()).unwrap());
        println!("{}", graph);
        graph.reset_nodes();
        println!("Length: {}", djikstra(graph.node_by_id("Siegburg").unwrap(), graph.node_by_id("Bergheim").unwrap()).unwrap_or(-1));
        println!("{}", graph);
    }

    #[test]
    fn djikstra_test() {
        let mut graph: Graph<char> = Graph::new();
        let node_a_idx = graph.add_node(Node::new('a'));
        let node_b_idx = graph.add_node(Node::new('b'));
        let node_c_idx = graph.add_node(Node::new('c'));
        let node_d_idx = graph.add_node(Node::new('d'));
        graph.add_edge(3, node_a_idx, node_b_idx);
        graph.add_edge(4, node_a_idx, node_c_idx);
        graph.add_edge(5, node_b_idx, node_a_idx);
        graph.add_edge(2, node_b_idx, node_d_idx);
        graph.add_edge(9, node_c_idx, node_a_idx);
        graph.add_edge(1, node_c_idx, node_d_idx);
        graph.add_edge(3, node_d_idx, node_b_idx);
        graph.add_edge(7, node_d_idx, node_c_idx);
        assert_eq!(5, djikstra(graph.node_by_id('a').unwrap(), graph.node_by_id('d').unwrap()).unwrap_or(-1));
        println!("Length: {}", djikstra(graph.node_by_id('a').unwrap(), graph.node_by_id('d').unwrap()).unwrap_or(-1));
        println!("{}", graph);
    }

    #[test]
    fn neighbor_positions_central() {
        let neighbors = neighbor_positions((2,2), 10, 10);
        assert_eq!((1, 2), neighbors[0]);
        assert_eq!((2, 1), neighbors[1]);
        assert_eq!((3, 2), neighbors[2]);
        assert_eq!((2, 3), neighbors[3]);
    }

    #[test]
    fn neighbor_positions_edge() {
        let neighbors = neighbor_positions((0,0), 10, 10);
        assert_eq!((1,0), neighbors[0]);
        assert_eq!((0,1), neighbors[1]);
        let neighbors = neighbor_positions((10, 10), 10, 10);
        assert_eq!((9, 10), neighbors[0]);
        assert_eq!((10, 9), neighbors[1]);
    }

    #[test]
    fn graph_from_vec() {
        let mut vec: Vec<Vec<i32>> = Vec::new();
        let vec_inner_1 = vec![3, 4, 5];
        let vec_inner_2 = vec![1, 2, 3];
        let vec_inner_3 = vec![1, 8, 2];
        vec.push(vec_inner_1);
        vec.push(vec_inner_2);
        vec.push(vec_inner_3);
        let graph = Graph::<String>::from_i32_vec(&vec);
        assert_eq!(8, djikstra(graph.node_by_id(String::from("[0|0]")).unwrap(), graph.node_by_id(String::from("[2|2]")).unwrap()).unwrap_or(-1));
        assert_eq!(7, djikstra(graph.node_by_id(String::from("[0|1]")).unwrap(), graph.node_by_id(String::from("[2|2]")).unwrap()).unwrap_or(-1));
    }

    #[test]
    fn big_vec() {
        let mut vec: Vec<Vec<i32>> = Vec::new();
        for i in 1..=130 {
            let mut inner_vec = Vec::new();
            for j in 1..=130 {
                inner_vec.push(i*j);
            }
            vec.push(inner_vec);
        }
        let graph = Graph::<String>::from_i32_vec(&vec);
        assert_eq!(8, djikstra(graph.node_by_id(String::from("[0|0]")).unwrap(), graph.node_by_id(String::from("[20|20]")).unwrap()).unwrap_or(-1));
    }
}
