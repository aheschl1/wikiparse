#!/usr/bin/env python3
"""
Example usage of the wikigraph module to visualize Wikipedia document links.
"""

from wikindex.wiki.wikigraph import visualize_document_links, get_graph_statistics
from wikindex.wiki.sqlite import Client

def main():
    # Database URL - adjust path if needed
    db_url = "sqlite:///wikindex_small.db"
    
    print("=== Wikipedia Document Links Graph Visualization ===\n")
    
    print("Creating visualization...")
    # Create visualization
    fig = visualize_document_links(
        db_url="sqlite:///wikindex_small.db",
        layout='spring',
        title="Wikipedia Document Links Network",
        node_size=8,
        show_labels=False  # Too many nodes for labels
    )
    
    # Save the visualization
    print("Saving visualization to 'wikipedia_links_graph.html'...")
    fig.write_html("wikipedia_links_graph.html")
    print("Graph saved! Open 'wikipedia_links_graph.html' in your browser to view it.")
    
    # Show graph statistics
    print("\n=== Graph Statistics ===")
    with Client(db_url) as client:
        from wikindex.wiki.wikigraph import fetch_document_links, create_networkx_graph, get_graph_statistics
        
        # Fetch links and create graph
        links = fetch_document_links(client, limit=1000)
        G = create_networkx_graph(links)
        
        # Get statistics
        stats = get_graph_statistics(G)
        
        print(f"Nodes: {stats['nodes']}")
        print(f"Edges: {stats['edges']}")
        print(f"Density: {stats['density']:.4f}")
        print(f"Is Connected: {stats['is_connected']}")
        
        if 'avg_degree' in stats:
            print(f"Average Degree: {stats['avg_degree']:.2f}")
            print(f"Max Degree: {stats['max_degree']}")
            print(f"Min Degree: {stats['min_degree']}")
        
        if 'avg_in_degree' in stats:
            print(f"Average In-Degree: {stats['avg_in_degree']:.2f}")
            print(f"Average Out-Degree: {stats['avg_out_degree']:.2f}")
    
    print("\n=== Most Connected Pages ===")
    with Client(db_url) as client:
        from wikindex.wiki.wikigraph import get_most_connected_nodes
        
        # Fetch links and create graph
        links = fetch_document_links(client, limit=1000)
        G = create_networkx_graph(links)
        
        # Get most connected nodes
        connected_nodes = get_most_connected_nodes(G, top_n=10)
        
        print("Most Connected (by total degree):")
        for i, (node, degree) in enumerate(connected_nodes['most_connected'][:5], 1):
            print(f"  {i}. {node} (degree: {degree})")
        
        if 'most_referenced' in connected_nodes:
            print("\nMost Referenced (by in-degree):")
            for i, (node, degree) in enumerate(connected_nodes['most_referenced'][:5], 1):
                print(f"  {i}. {node} (in-degree: {degree})")

if __name__ == "__main__":
    main()
