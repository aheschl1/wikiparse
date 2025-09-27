from wikindex.wiki.sqlite import Client, Document, document_links
import plotly.graph_objects as go
import networkx as nx
from sqlalchemy import text
import statistics


def fetch_document_links(client: Client, limit=None):
    """
    Fetch all document links from the database.
    
    Args:
        client: Database client instance
        limit: Optional limit on number of links to fetch (for large datasets)
    
    Returns:
        List of tuples (from_title, to_title)
    """
    query = text("SELECT from_title, to_title FROM document_links")
    if limit:
        query = text(f"SELECT from_title, to_title FROM document_links LIMIT {limit}")
    
    result = client.session.execute(query)
    return [(row.from_title, row.to_title) for row in result]


def create_networkx_graph(links):
    """
    Create a NetworkX graph from document links.
    
    Args:
        links: List of tuples (from_title, to_title)
    
    Returns:
        NetworkX DiGraph object
    """
    G = nx.DiGraph()
    
    # Add edges (nodes will be added automatically)
    for from_title, to_title in links:
        G.add_edge(from_title, to_title)
    
    return G


def calculate_graph_positions(G, layout='random', **layout_kwargs):
    """
    Calculate node positions for visualization using various layout algorithms.
    
    Args:
        G: NetworkX graph
        layout: Layout algorithm ('random', 'circular', 'shell', 'spring')
        **layout_kwargs: Additional arguments for the layout algorithm
    
    Returns:
        Dictionary of node positions {node: (x, y)}
    """
    # Set default scaling for more spread out graphs
    default_scale = layout_kwargs.pop('scale', 5.0)  # Much larger scale
    
    if layout == 'random':
        pos = nx.random_layout(G, **layout_kwargs)
    elif layout == 'circular':
        pos = nx.circular_layout(G, **layout_kwargs)
    elif layout == 'shell':
        pos = nx.shell_layout(G, **layout_kwargs)  
    elif layout == 'spring':
        # Try spring layout but fall back to random if it fails
        try:
            # Use more iterations and larger k for better separation
            pos = nx.spring_layout(G, k=2.0, iterations=100, **layout_kwargs)
        except:
            print("Spring layout failed, falling back to random layout")
            pos = nx.random_layout(G, **layout_kwargs)
    else:
        # Default to random layout
        pos = nx.random_layout(G, **layout_kwargs)
    
    # Scale positions to spread them out more
    scaled_pos = {}
    for node, (x, y) in pos.items():
        scaled_pos[node] = (x * default_scale, y * default_scale)
    
    return scaled_pos


def create_plotly_graph(G, pos=None, title="Wikipedia Document Links Graph", 
                       node_size=5, edge_width=1, show_labels=True, 
                       max_labels=100, width=1600, height=1200):
    """
    Create a Plotly visualization of the NetworkX graph.
    
    Args:
        G: NetworkX graph
        pos: Node positions dictionary (if None, will calculate using spring layout)
        title: Graph title
        node_size: Size of nodes
        edge_width: Width of edges
        show_labels: Whether to show node labels
        max_labels: Maximum number of labels to show (to avoid clutter)
        width: Width of the plot in pixels
        height: Height of the plot in pixels
    
    Returns:
        Plotly Figure object
    """
    if pos is None:
        pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Extract edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=edge_width, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Extract nodes
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    
    nodes = list(G.nodes())
    for node in nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Calculate node degree for hover info
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        total_degree = in_degree + out_degree
        
        node_info.append(f'{node}<br>In-degree: {in_degree}<br>Out-degree: {out_degree}<br>Total degree: {total_degree}')
        
        # Only show labels for nodes with highest degrees if we have too many
        if show_labels and len(nodes) <= max_labels:
            node_text.append(node)
        elif show_labels and total_degree >= sorted([G.in_degree(n) + G.out_degree(n) for n in nodes], reverse=True)[min(max_labels-1, len(nodes)-1)]:
            node_text.append(node)
        else:
            node_text.append('')
    
    # Calculate node degrees for coloring
    node_degrees = [G.in_degree(node) + G.out_degree(node) for node in nodes]
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text' if show_labels else 'markers',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        hovertext=node_info,
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            reversescale=True,
            color=node_degrees,
            size=node_size,
            colorbar=dict(
                thickness=15,
                len=0.5,
                x=1.05,
                title="Node Degree"
            ),
            line=dict(width=2)
        )
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(text=title, font=dict(size=16)),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ 
                            dict(
                                text=f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002,
                                xanchor='left', yanchor='bottom',
                                font=dict(size=12)
                            )
                        ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        width=width,
                        height=height,
                        # Enable zoom and pan
                        dragmode='pan'
                    ))
    
    return fig


def visualize_document_links(db_url="sqlite:///wikindex.db", limit=None, 
                           layout='random', title="Wikipedia Document Links Graph",
                           node_size=5, edge_width=1, show_labels=True, 
                           max_labels=100, width=3200, height=2400):
    """
    Complete function to fetch data and create visualization.
    
    Args:
        db_url: Database URL
        limit: Optional limit on number of links to fetch
        layout: Layout algorithm for positioning nodes
        title: Graph title
        node_size: Size of nodes
        edge_width: Width of edges
        show_labels: Whether to show node labels
        max_labels: Maximum number of labels to show
        width: Width of the plot in pixels (default 3200 for wide viewing)
        height: Height of the plot in pixels (default 2400 for tall viewing)
    
    Returns:
        Plotly Figure object
    """
    with Client(db_url) as client:
        # Fetch links
        print("Fetching document links...")
        links = fetch_document_links(client, limit=limit)
        print(f"Found {len(links)} links")
        
        if not links:
            print("No links found in database")
            return None
        
        # Create graph
        print("Creating NetworkX graph...")
        G = create_networkx_graph(links)
        print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Calculate positions
        print(f"Calculating node positions using {layout} layout...")
        pos = calculate_graph_positions(G, layout=layout, scale=10.0)
        
        # Create visualization
        print("Creating Plotly visualization...")
        fig = create_plotly_graph(G, pos=pos, title=title, 
                                 node_size=node_size, edge_width=edge_width,
                                 show_labels=show_labels, max_labels=max_labels,
                                 width=width, height=height)
        
        return fig


def get_graph_statistics(G):
    """
    Get basic statistics about the graph.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Dictionary with graph statistics
    """
    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_connected': nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G),
    }
    
    # Calculate degree statistics
    degrees = [d for n, d in G.degree()]
    if degrees:
        stats['avg_degree'] = sum(degrees) / len(degrees)
        stats['max_degree'] = max(degrees)
        stats['min_degree'] = min(degrees)
    
    # For directed graphs, also calculate in/out degree stats
    if G.is_directed():
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        
        if in_degrees:
            stats['avg_in_degree'] = sum(in_degrees) / len(in_degrees)
            stats['max_in_degree'] = max(in_degrees)
            
        if out_degrees:
            stats['avg_out_degree'] = sum(out_degrees) / len(out_degrees)
            stats['max_out_degree'] = max(out_degrees)
    
    return stats


def get_most_connected_nodes(G, top_n=10):
    """
    Get the most connected nodes in the graph.
    
    Args:
        G: NetworkX graph
        top_n: Number of top nodes to return
    
    Returns:
        Dictionary with lists of most connected nodes by different metrics
    """
    result = {}
    
    # Overall degree (in + out for directed graphs)
    degrees = [(node, G.in_degree(node) + G.out_degree(node)) for node in G.nodes()]
    result['most_connected'] = sorted(degrees, key=lambda x: x[1], reverse=True)[:top_n]
    
    if G.is_directed():
        # Most linked-to pages (highest in-degree)
        in_degrees = [(node, G.in_degree(node)) for node in G.nodes()]
        result['most_referenced'] = sorted(in_degrees, key=lambda x: x[1], reverse=True)[:top_n]
        
        # Pages that link to most others (highest out-degree)
        out_degrees = [(node, G.out_degree(node)) for node in G.nodes()]
        result['most_linking'] = sorted(out_degrees, key=lambda x: x[1], reverse=True)[:top_n]
    
    return result


def create_subgraph_around_node(G, center_node, radius=1):
    """
    Create a subgraph centered around a specific node.
    
    Args:
        G: NetworkX graph
        center_node: The node to center the subgraph around
        radius: Number of hops to include from the center node
    
    Returns:
        NetworkX graph (subgraph)
    """
    if center_node not in G:
        raise ValueError(f"Node '{center_node}' not found in graph")
    
    # Get all nodes within radius hops
    nodes_to_include = {center_node}
    
    for r in range(radius):
        new_nodes = set()
        for node in nodes_to_include:
            # Add neighbors (both incoming and outgoing for directed graphs)
            new_nodes.update(G.neighbors(node))
            if G.is_directed():
                new_nodes.update(G.predecessors(node))
        nodes_to_include.update(new_nodes)
    
    return G.subgraph(nodes_to_include).copy()