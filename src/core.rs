use std::{fmt::Display, rc::Rc, cell::RefCell, collections::HashMap};

/// An edge between two nodes inside the graph
#[derive(Clone, Eq)]
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
#[derive(Clone, Eq)]
pub struct Node<T: Display> {
    pub id: T,
    pub edges: Vec<Edge<T>>,
    pub distance: i32,
    pub shortest_path: Vec<Rc<RefCell<Node<T>>>>,
}

impl<T: Display + Eq + Clone> Node<T> {
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
    pub fn add_edge(&mut self, edge: Edge<T>) {
        self.edges.push(edge)
    }

    /// Returns the shortest path to this node.
    /// 
    /// For a node to receive its shortest path a path finding algorithm has to have run beforehand.
    /// 
    /// # Example
    /// ```
    /// use lmh01_pathfinding::{core::{Graph, Node}, algorithms::dijkstra};
    /// 
    /// // Prepare graph
    /// let mut graph: Graph<char> = Graph::new();
    /// let node_a_idx = graph.add_node(Node::new('a'));
    /// let node_b_idx = graph.add_node(Node::new('b'));
    /// let node_c_idx = graph.add_node(Node::new('c'));
    /// let node_d_idx = graph.add_node(Node::new('d'));
    /// graph.add_edge(3, node_a_idx, node_b_idx);
    /// graph.add_edge(4, node_a_idx, node_c_idx);
    /// graph.add_edge(3, node_b_idx, node_a_idx);
    /// graph.add_edge(2, node_b_idx, node_d_idx);
    /// graph.add_edge(9, node_c_idx, node_a_idx);
    /// graph.add_edge(1, node_c_idx, node_d_idx);
    /// graph.add_edge(3, node_d_idx, node_b_idx);
    /// graph.add_edge(7, node_d_idx, node_c_idx);
    /// dijkstra(graph.node_by_id(&'a').unwrap(), graph.node_by_id(&'d').unwrap()).unwrap_or(-1);
    /// 
    /// // Get shortest path
    /// let string = graph.node_by_id(&'d').unwrap().borrow_mut().shortest_path();
    /// assert_eq!("a -> b -> d", string)
    /// ```
    pub fn shortest_path(&self) -> String {
        let mut path: Vec<T> = Vec::new();
        for previous in &self.shortest_path {
            path.push(previous.borrow().id.clone());
        }
        let mut path_string = String::new();
        for previous in path {
            path_string.push_str(&format!("{} -> ", previous));
        }
        path_string.push_str(&format!("{}", self.id));
        path_string
    }

    /// Returns the distance to this node.
    /// 
    /// For a node to receive its distance a path finding algorithm has to have run beforehand.
    pub fn distance(&self) -> i32 {
        self.distance
    }
}

impl<T: Display + Eq> PartialEq for Node<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T: Display + Eq> Display for Node<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id)
    }
}

impl<T: Display + Eq> PartialOrd for Node<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.distance.cmp(&other.distance).reverse())
    }
}

impl<T: Display + Eq> Ord for Node<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.cmp(&other.distance).reverse()
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
    pub fn node_by_id(&self, id: &T) -> Option<Rc<RefCell<Node<T>>>> {
        for node in self.nodes.iter() {
            if node.borrow().id.eq(id) {
                return Some(node.clone());
            }
        }
        None
    }


    /// Resets the distance of each node in the graph back to `i32::MAX` and resets the shortest path string.
    /// 
    /// Should be called after [djikstra](./fn.djikstra.html) has run, otherwise the next call might not result in the correct distance.
    pub fn reset_nodes(&mut self) {
        for node in self.nodes.iter_mut() {
            node.borrow_mut().distance = i32::MAX;
            node.borrow_mut().shortest_path = Vec::new();
        }
    }

    /// Prints the shortest path to the target node.
    /// 
    /// Requires that [djikstra](./fn.djikstra.html) has run to fill the shortest paths.
    /// 
    /// Uses [Node::shortest_path()](./struct.Node.html#method.shortest_path) to print the shortest path.
    /// 
    /// Use [Node::shortest_path()](./struct.Node.html#method.shortest_path) instead if you would like to receive the shortest path as string.
    pub fn print_shortest_path(&self, target_node_id: T) {
        let node = self.node_by_id(&target_node_id).unwrap();
        println!("Shortest path to {}: {}", node.borrow().id, node.borrow().shortest_path());
    }

    /// Create a graph from a 2D vector containing i32.
    /// 
    /// The i32 value is the edge weight of each edge leading into that node.
    /// # Example
    /// ```
    /// use lmh01_pathfinding::{core::Graph, algorithms::dijkstra};
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
    /// // Run dijkstra's algorithm
    /// assert_eq!(8, dijkstra(graph.node_by_id(&String::from("[0|0]")).unwrap(), graph.node_by_id(&String::from("[2|2]")).unwrap()).unwrap_or(-1));
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

    /// Constructs a graph from a list of instructions. This is meant to be used by reading the instructions form a file.
    /// 
    /// The order in which the instructions are stored in the vector does not matter.
    /// 
    /// # Instructions
    /// 
    /// ## Nodes
    /// 
    /// ```txt
    /// node: LABEL1
    /// ```
    /// This declares a new node labeled `LABEL1`
    /// 
    /// ## Edge
    /// 
    /// ```txt
    /// edge: LABEL1 WEIGHT LABEL2
    /// ```
    /// This adds an edge from `LABEL1` to `LABEL2` with `WEIGHT`
    /// 
    /// ```txt
    /// double_edge: LABEL1 WEIGHT LABEL2
    /// ```
    /// This adds a double edge between `LABEL1` and `LABEL2` with `WEIGHT`
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use lmh01_pathfinding::algorithms::dijkstra;
    /// use lmh01_pathfinding::core::Graph;
    /// 
    /// // This lines vector should ideally constructed by parsing a file, below insertions are just for demonstration.
    /// let mut lines = Vec::new();
    /// lines.push(String::from("node: a"));
    /// lines.push(String::from("node: b"));
    /// lines.push(String::from("node: c"));
    /// lines.push(String::from("node: d"));
    /// lines.push(String::from("edge: a 7 b"));
    /// lines.push(String::from("edge: a 4 c"));
    /// lines.push(String::from("edge: b 2 d"));
    /// lines.push(String::from("edge: c 9 d"));
    /// lines.push(String::from("edge: c 2 b"));
    /// lines.push(String::from("double_edge: a 1 d"));
    /// let graph = Graph::<String>::from_instructions(&lines);
    /// assert_eq!(1, dijkstra(graph.node_by_id(&String::from("a")).unwrap(), graph.node_by_id(&String::from("d")).unwrap()).unwrap_or(-1));
    /// ```
    pub fn from_instructions(instructions: &Vec<String>) -> Graph<String> {
        // Stores all node labels of nodes that should be added to the graph
        let mut node_labels = Vec::new();
        // Stores all edges that should be added to the graph, (WEIGHT, LABEL1, LABEL2, double)
        let mut edges: Vec<(i32, String, String, bool)> = Vec::new();

        // Parse lines
        for line in instructions {
            let split: Vec<&str> = line.split(' ').collect();
            match split[0].to_lowercase().as_str() {
                "node:" => {
                    node_labels.push(String::from(split[1]));
                },
                "edge:" => {
                    edges.push((split[2].parse::<i32>().expect("Unable to parse edge weight!"), String::from(split[1]), String::from(split[3]), false));
                },
                "double_edge:" => {
                    edges.push((split[2].parse::<i32>().expect("Unable to parse edge weight!"), String::from(split[1]), String::from(split[3]), true));
                },
                _ => (),
            }
        }

        let mut graph = Graph::new();

        // Because node indexes are required to add edges to the graph, this map stores node labels mapped to the index
        let mut node_indexes = HashMap::new();
        // Add nodes to graph
        for label in node_labels {
            let idx = graph.add_node(Node::new(label.clone()));
            node_indexes.insert(label, idx);
        }
        // Add edges to graph
        for edge in edges {
            if edge.3 {
                graph.add_double_edge(edge.0,*node_indexes.get(&edge.1).expect("Unable to find edge index!"), *node_indexes.get(&edge.2).expect("Unable to find edge index!"));
            } else {
                graph.add_edge(edge.0,*node_indexes.get(&edge.1).expect("Unable to find edge index!"), *node_indexes.get(&edge.2).expect("Unable to find edge index!"));
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

/// Returns the neighboring positions for a position in a 2D graph.
/// 
/// # Example
/// ```
/// use lmh01_pathfinding::core::neighbor_positions;
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

mod tests {
    use std::{collections::BinaryHeap, rc::Rc, cell::RefCell};

    use crate::{core::{neighbor_positions, Graph, Node}, algorithms::dijkstra};


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
        assert_eq!(8, dijkstra(graph.node_by_id(&String::from("[0|0]")).unwrap(), graph.node_by_id(&String::from("[2|2]")).unwrap()).unwrap_or(-1));
        graph.print_shortest_path(String::from("[2|2]"));
        assert_eq!(7, dijkstra(graph.node_by_id(&String::from("[0|1]")).unwrap(), graph.node_by_id(&String::from("[2|2]")).unwrap()).unwrap_or(-1));
    }

    #[test]
    fn graph_from_instructions() {
        let mut lines = Vec::new();
        lines.push(String::from("node: a"));
        lines.push(String::from("node: b"));
        lines.push(String::from("node: c"));
        lines.push(String::from("node: d"));
        lines.push(String::from("edge: a 7 b"));
        lines.push(String::from("edge: a 4 c"));
        lines.push(String::from("edge: b 2 d"));
        lines.push(String::from("edge: c 9 d"));
        lines.push(String::from("edge: c 2 b"));
        let graph = Graph::<String>::from_instructions(&lines);
        assert_eq!(8, dijkstra(graph.node_by_id(&String::from("a")).unwrap(), graph.node_by_id(&String::from("d")).unwrap()).unwrap_or(-1));
    }

    #[test]
    fn graph_from_instructions_2() {
        let mut lines = Vec::new();
        lines.push(String::from("node: a"));
        lines.push(String::from("node: b"));
        lines.push(String::from("node: c"));
        lines.push(String::from("node: d"));
        lines.push(String::from("edge: a 3 b"));
        lines.push(String::from("edge: b 5 a"));
        lines.push(String::from("edge: a 1 c"));
        lines.push(String::from("edge: c 9 a"));
        lines.push(String::from("edge: b 1 d"));
        lines.push(String::from("edge: d 3 b"));
        lines.push(String::from("edge: c 3 d"));
        lines.push(String::from("edge: d 7 c"));
        lines.push(String::from("edge: c 1 b"));
        let graph = Graph::<String>::from_instructions(&lines);
        assert_eq!(3, dijkstra(graph.node_by_id(&String::from("a")).unwrap(), graph.node_by_id(&String::from("d")).unwrap()).unwrap_or(-1));
    }

    #[test]
    fn graph_from_instructions_3() {
        let mut lines = Vec::new();
        lines.push(String::from("node: a"));
        lines.push(String::from("node: b"));
        lines.push(String::from("node: c"));
        lines.push(String::from("node: d"));
        lines.push(String::from("edge: a 3 b"));
        lines.push(String::from("edge: b 5 a"));
        lines.push(String::from("edge: a 1 c"));
        lines.push(String::from("edge: c 9 a"));
        lines.push(String::from("edge: b 1 d"));
        lines.push(String::from("edge: d 3 b"));
        lines.push(String::from("edge: c 3 d"));
        lines.push(String::from("edge: d 7 c"));
        lines.push(String::from("edge: c 1 b"));
        lines.push(String::from("double_edge: a 1 d"));
        let graph = Graph::<String>::from_instructions(&lines);
        println!("{graph}");
        assert_eq!(1, dijkstra(graph.node_by_id(&String::from("a")).unwrap(), graph.node_by_id(&String::from("d")).unwrap()).unwrap_or(-1));
    }

    #[test]
    fn node_shortest_path() {
        // Prepare graph
        let mut graph: Graph<char> = Graph::new();
        let node_a_idx = graph.add_node(Node::new('a'));
        let node_b_idx = graph.add_node(Node::new('b'));
        let node_c_idx = graph.add_node(Node::new('c'));
        let node_d_idx = graph.add_node(Node::new('d'));
        let node_e_idx = graph.add_node(Node::new('e'));
        graph.add_edge(3, node_a_idx, node_b_idx);
        graph.add_edge(4, node_a_idx, node_c_idx);
        graph.add_edge(3, node_b_idx, node_a_idx);
        graph.add_edge(2, node_b_idx, node_d_idx);
        graph.add_edge(9, node_c_idx, node_a_idx);
        graph.add_edge(1, node_c_idx, node_d_idx);
        graph.add_edge(3, node_d_idx, node_b_idx);
        graph.add_edge(7, node_d_idx, node_c_idx);
        graph.add_edge(8, node_d_idx, node_e_idx);
        dijkstra(graph.node_by_id(&'a').unwrap(), graph.node_by_id(&'e').unwrap()).unwrap_or(-1);

        // Get shortest path
        let string = graph.node_by_id(&'e').unwrap().borrow_mut().shortest_path();
        assert_eq!("a -> b -> d -> e", string)
    }

    #[test]
    fn binary_heap_test() {
        let mut node_1 = Node::new(1);
        node_1.distance = 5;
        let mut node_2 = Node::new(2);
        node_2.distance = 4;
        let mut node_3 = Node::new(3);
        node_3.distance = 9;
        let mut open_nodes: BinaryHeap<Rc<RefCell<Node<i32>>>> = BinaryHeap::new();
        open_nodes.push(Rc::new(RefCell::new(node_1)));
        open_nodes.push(Rc::new(RefCell::new(node_2)));
        open_nodes.push(Rc::new(RefCell::new(node_3)));
        assert_eq!(4, open_nodes.pop().unwrap().borrow_mut().distance);
        assert_eq!(5, open_nodes.pop().unwrap().borrow_mut().distance);
        assert_eq!(9, open_nodes.pop().unwrap().borrow_mut().distance);
    }

    #[test]
    fn big_vec() {
        let mut vec: Vec<Vec<i32>> = Vec::new();
        for i in 1..=100 {
            let mut inner_vec = Vec::new();
            for j in 1..=100 {
                inner_vec.push(i*j);
            }
            vec.push(inner_vec);
        }
        let graph = Graph::<String>::from_i32_vec(&vec);
        assert_eq!(5060, dijkstra(graph.node_by_id(&String::from("[0|0]")).unwrap(), graph.node_by_id(&String::from("[20|20]")).unwrap()).unwrap_or(-1));
    }
}