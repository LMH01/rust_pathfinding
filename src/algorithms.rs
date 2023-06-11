use std::{fmt::Display, hash::Hash, rc::Rc, cell::RefCell, collections::{BinaryHeap, HashSet}};

use crate::core::{Node, Graph};

/// Calculates the shortest distance between two nodes.
/// This will utilize the algorithm by [djikstra](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm).
///
/// This algorithm does not work properly on graphs with negative edge weights.
///  
/// The distance field in each node should be set to `i32:MAX` before this function is called.
/// When the nodes are organized using the [Graph](struct.Graph.html) struct the function [reset_nodes](struct.Graph.html#method.reset_nodes) may be used to reset the distance field.
/// # Params
/// `graph` - the graph on which the algorithm should be run
/// 
/// `source` - id of the source node
/// 
/// `target` - id of the target node
/// # Returns
/// `Some(length)` when the shortest path was found.
/// 
/// `None` when no path between the two nodes exists.
/// 
/// # Examples
/// ```rust
/// use lmh01_pathfinding::{core::{Node, Graph}, algorithms::dijkstra};
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
/// let result = dijkstra(&mut graph, &'a', &'d').unwrap_or(-1);
/// assert_eq!(5, result);
/// 
/// // Reset node distances before running the algorithm again
/// graph.reset_nodes();
/// 
/// // Run algorithm again
/// let result = dijkstra(&mut graph, &'b', &'c').unwrap_or(-1);
/// assert_eq!(9, result);
/// 
/// // Reset nodes again
/// 
/// // Run algorithm again, returns -1 because no node exists that connects e to the rest of the graph.
/// let result = dijkstra(&mut graph, &'a', &'e').unwrap_or(-1);
/// assert_eq!(-1, result);
/// 
/// ```
/// It is also possible to create a graph from a vector. For more information take a look [here](struct.Graph.html#method.from_i32_vec).
pub fn dijkstra<T: Display + Clone + Eq + Hash>(graph: &mut Graph<T>, source_node_id: &T, target_node_id: &T) -> Option<i32> {
    graph.reset_nodes();
    let source_node = graph.node_by_id(source_node_id)?;
    source_node.borrow_mut().distance = 0;
    let mut open_nodes: BinaryHeap<Rc<RefCell<Node<T>>>> = BinaryHeap::new();
    let mut open_node_ids: HashSet<T> = HashSet::new();
    let mut closed_node_ids: HashSet<T> = HashSet::new();
    //let mut closed_nodes: Vec<Rc<RefCell<Node<T>>>> = Vec::new();
    open_nodes.push(source_node.clone());

    while !open_nodes.is_empty() {
        let node = open_nodes.pop().unwrap();

        for edge in &node.borrow().edges {
            let target = &edge.target;
            let edge_weight = edge.weight;
            if !closed_node_ids.contains(&target.borrow().id) {
                let new_distance = node.borrow().distance + edge_weight;
                calc_min_distance(target, edge_weight, &node);
                if new_distance < target.borrow().distance {
                    target.borrow_mut().distance = new_distance;
                }
                if !open_node_ids.contains(&target.borrow().id) {
                    open_nodes.push(target.clone());
                    open_node_ids.insert(target.borrow().clone().id);
                }
            }
        }
        closed_node_ids.insert(node.borrow().clone().id);
        if cfg!(feature = "steps") {
            println!("Last node: {}, distance: {:2} | open: {:12}, closed: {:12}", node.borrow().id, node.borrow().distance, open_nodes.len(), closed_node_ids.len());
        }
    }

    let target_distance = graph.node_by_id(target_node_id)?.borrow().distance;
    if target_distance == i32::MAX {
        None
    } else {
        Some(target_distance)
    }
}

fn calc_min_distance<T: Display + Eq + Clone>(node: &Rc<RefCell<Node<T>>>, weight: i32, source: &Rc<RefCell<Node<T>>>) {
    let source_distance = source.borrow().distance;
    if source_distance + weight < node.borrow().distance {
        node.borrow_mut().distance = source_distance + weight;
        let mut shortest_path = source.borrow().shortest_path.clone();
        shortest_path.push(source.clone());
        node.borrow_mut().shortest_path = shortest_path;
    }
}

/// Calculates the shortest distance between two nodes.
/// This will utilize the [Bellman-Ford algorithm](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm).
/// 
/// This algorithm works on graphs with negative edge weights but is slower than Dijkstra's algorithm.
/// 
/// # Params
/// `graph` - the graph on which the algorithm should be run
/// 
/// `source` - id of the source node
/// 
/// `target` - id of the target node
/// # Returns
/// `Some(length)` when the shortest path was found.
/// 
/// `None` when no path between the two nodes exists.
/// 
/// # Examples
/// ```rust
/// use lmh01_pathfinding::{core::{Node, Graph}, algorithms::bellman_ford};
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
/// // Run Bellman-Ford algorithm to determine the shortest path, result contains the shortest distance.
/// let result = bellman_ford(&mut graph, &'a', &'d').unwrap_or(-1);
/// assert_eq!(5, result);
/// 
/// let result = bellman_ford(&mut graph, &'b', &'c').unwrap_or(-1);
/// assert_eq!(9, result);
/// 
/// // Run algorithm again, returns -1 because no node exists that connects e to the rest of the graph.
/// let result = bellman_ford(&mut graph, &'a', &'e').unwrap_or(-1);
/// assert_eq!(-1, result);
/// 
/// ```
/// Parts of this function where created using ChatGPT
pub fn bellman_ford<T: Display + Eq + Clone>(graph: &mut Graph<T>, source_node_id: &T, target_node_id: &T) -> Option<i32> {
    graph.reset_nodes();

    let source_node = graph.node_by_id(source_node_id)?;
    source_node.borrow_mut().distance = 0;

    let nodes_count = graph.size();

    for _ in 0..nodes_count - 1 {
        for node in graph.nodes() {
            let node_ref = node.borrow();

            if node_ref.distance == std::i32::MAX {
                continue;
            }

            for edge in &node_ref.edges {
                let target_node = edge.target.clone();
                let new_distance = node_ref.distance + edge.weight;

                if new_distance < target_node.borrow().distance {
                    target_node.borrow_mut().distance = new_distance;
                    let mut shortest_path = node_ref.shortest_path.clone();
                    shortest_path.push(Rc::clone(node));
                    target_node.borrow_mut().shortest_path = shortest_path;
                }
            }
        }
    }

    let target_node = graph.node_by_id(target_node_id)?;
    let target_distance = target_node.borrow().distance;

    if target_distance == std::i32::MAX {
        None
    } else {
        Some(target_distance)
    }
}

mod tests {
    use crate::{core::{Graph, Node}, algorithms::{self, dijkstra, bellman_ford}};

    #[test]
    fn dijkstra_test_1() {
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
        println!("Length: {}", dijkstra(&mut graph, &"Siegburg", &"Bergheim").unwrap_or(-1));
        println!("{}", graph);
    }

    #[test]
    fn dijkstra_test_2() {
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
        assert_eq!(5, dijkstra(&mut graph, &'a', &'d').unwrap_or(-1));
        println!("Length: {}", dijkstra(&mut graph, &'a', &'d').unwrap_or(-1));
        println!("{}", graph);
    }

    #[test]
    fn bellman_ford_test_1() {
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
        println!("{graph}");
        assert_eq!(8, bellman_ford(&mut graph, &"Siegburg", &"Troisdorf").unwrap_or(-1));
        assert_eq!(18, bellman_ford(&mut graph, &"Siegburg", &"Bergheim").unwrap_or(-1));
    }

    #[test]
    fn bellman_ford_test_2() {
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
        assert_eq!(5, bellman_ford(&mut graph, &'a', &'d').unwrap_or(-1));
    }

    #[test]
    fn bellman_ford_test_negative() {
        let mut graph: Graph<char> = Graph::new();
        let node_a_idx = graph.add_node(Node::new('a'));
        let node_b_idx = graph.add_node(Node::new('b'));
        let node_c_idx = graph.add_node(Node::new('c'));
        let node_d_idx = graph.add_node(Node::new('d'));
        graph.add_double_edge(4, node_a_idx, node_b_idx);
        graph.add_edge(2, node_a_idx, node_c_idx);
        graph.add_edge(-1, node_c_idx, node_a_idx);
        graph.add_edge(5, node_a_idx, node_d_idx);
        graph.add_edge(-3, node_d_idx, node_b_idx);
        graph.add_double_edge(2, node_b_idx, node_c_idx);
        graph.add_edge(7, node_d_idx, node_c_idx);
        println!("{graph}");
        assert_eq!(1, bellman_ford(&mut graph, &'c', &'b').unwrap_or(-1));
        assert_eq!(-3, bellman_ford(&mut graph, &'d', &'b').unwrap_or(-1));
        assert_eq!(-2, bellman_ford(&mut graph, &'d', &'a').unwrap_or(-1));
    }
    
}