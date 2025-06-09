import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import hashlib # For generating consistent colors
import os # For path operations
import matplotlib.colors # For ListedColormap and luminance calculation

def get_color_for_id(node_id, chosen_cmap_name='Pastel1'):
    """
    Generates a consistent color based on node_id using a chosen colormap.

    Args:
        node_id: The ID to generate a color for.
        chosen_cmap_name (str): Name of the Matplotlib colormap to use (e.g., 'Pastel1', 'Set3').

    Returns:
        A color (RGBA tuple).
    """
    try:
        cmap = plt.cm.get_cmap(chosen_cmap_name)
    except ValueError:
        print(f"Warning: Colormap '{chosen_cmap_name}' not found. Defaulting to 'Pastel1'.")
        cmap = plt.cm.get_cmap('Pastel1')
        
    # Use a hash of the node_id to get a consistent index
    hash_object = hashlib.md5(str(node_id).encode())
    hex_dig = hash_object.hexdigest()
    # Use modulo of cmap.N to ensure index is within the colormap's defined colors
    color_index = int(hex_dig, 16) % cmap.N
    return cmap(color_index)

def get_text_color_for_background(bg_color_rgba):
    """
    Determines if text should be black or white based on background color's luminance.
    Args:
        bg_color_rgba (tuple): Background color (R, G, B, A).
    Returns:
        'black' or 'white'.
    """
    r, g, b = bg_color_rgba[:3] # Ignore alpha for luminance calculation
    # Standard luminance calculation
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'black' if luminance > 0.5 else 'white'


def visualize_computation_graph(reassigned_id_graph, stages, reassigned_id_to_node_id=None,
                                title="Computation Graph (Reassigned IDs)",
                                output_dir=None, output_filename="computation_graph.png",
                                stage_cmap_name='Pastel1'):
    """
    Visualizes the reassigned_id_graph, attempting a sequential layout (left-to-right).
    Colors nodes by stage with subtle colors. Saves the plot if output_dir is specified.
    (Args and other parts of the docstring remain the same)
    """
    fig = plt.figure(figsize=(16, 10)) # Adjusted figure size for potentially wider layout
    G = nx.DiGraph()
    node_labels = {}
    node_to_stage = {}

    # ... (rest of the node and stage processing logic from the previous version remains the same) ...
    for stage_idx, stage_data in enumerate(stages):
        nodes_in_stage = stage_data[1]
        for reassigned_id in nodes_in_stage:
            node_to_stage[reassigned_id] = stage_idx

    max_reassigned_id = -1
    if reassigned_id_graph:
        for i, adj in enumerate(reassigned_id_graph):
            max_reassigned_id = max(max_reassigned_id, i)
            if G.has_node(i): # Ensure node i exists
                 pass # Already added or will be by all_known_reassigned_ids
            else:
                 G.add_node(i)
            for neighbor in adj:
                max_reassigned_id = max(max_reassigned_id, neighbor)
                if G.has_node(neighbor):
                    pass
                else:
                    G.add_node(neighbor)
                G.add_edge(i, neighbor)
    
    all_nodes_in_stages = set(n_id for s_idx, s_data in enumerate(stages) for n_id in s_data[1])
    all_graph_nodes = set(range(max_reassigned_id + 1)) if max_reassigned_id != -1 else set()
    all_known_reassigned_ids = all_nodes_in_stages.union(all_graph_nodes)

    if not G.nodes() and not all_known_reassigned_ids:
        print("Warning: Computation graph is empty. No visualization will be generated.")
        plt.close(fig)
        if output_dir: # Clean up empty figure file if it was somehow created
            save_file_path = os.path.join(output_dir, output_filename)
            if os.path.exists(save_file_path) and os.path.getsize(save_file_path) < 1024: # Heuristic for empty plot
                 os.remove(save_file_path)
        return

    for node_id in all_known_reassigned_ids:
        if not G.has_node(node_id): # Add nodes that might be in stages but not graph edges
            G.add_node(node_id)
        label = str(node_id)
        if reassigned_id_to_node_id and node_id < len(reassigned_id_to_node_id):
            original_id = reassigned_id_to_node_id[node_id]
            label = f"R{node_id}\n(ONNX {original_id})"
        elif reassigned_id_to_node_id is None:
            label = f"R{node_id}"
        node_labels[node_id] = label
    
    # Ensure all nodes in node_labels are in G, especially if G was initially empty
    for node_id_in_label in node_labels.keys():
        if not G.has_node(node_id_in_label):
            G.add_node(node_id_in_label)


    node_colors = []
    actual_stage_indices_in_use = sorted(list(set(s_idx for s_idx in node_to_stage.values() if s_idx is not None)))
    stage_color_map = {stage_idx: get_color_for_id(stage_idx, chosen_cmap_name=stage_cmap_name)
                       for stage_idx in actual_stage_indices_in_use}

    for node in G.nodes():
        stage_idx = node_to_stage.get(node, -1)
        if stage_idx != -1 and stage_idx in stage_color_map:
            node_colors.append(stage_color_map[stage_idx])
        else:
            node_colors.append('lightgrey')

    # Attempt sequential layout
    pos = None
    layout_method = "spring" # Default if all others fail

    if G.number_of_nodes() > 0 : # Only attempt layout if there are nodes
        try:
            # Try Graphviz 'dot' layout with Left-to-Right ranking
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR')
            layout_method = "Graphviz dot (LR)"
            print(f"Layout method: {layout_method}")
        except Exception as e_dot:
            print(f"Graphviz layout (prog='dot' with LR rankdir) failed: {e_dot}. Trying multipartite layout.")
            try:
                # Fallback to multipartite_layout if graph is a DAG
                node_layers = {}
                # Compute layers using topological generations
                for i, layer_nodes_gen in enumerate(nx.topological_generations(G)):
                    for node_in_gen in layer_nodes_gen:
                        node_layers[node_in_gen] = i
                
                # Assign layer attribute to nodes for multipartite_layout
                for node, layer_val in node_layers.items():
                    G.nodes[node]['layer_attr'] = layer_val
                
                # Check if all nodes got a layer, important for disconnected components.
                # For multipartite, all nodes need the subset_key attribute.
                max_layer_val = max(node_layers.values()) if node_layers else -1
                for node in G.nodes():
                    if 'layer_attr' not in G.nodes[node]:
                        # Place unlayered nodes (e.g. part of a cycle if topological_generations only processed DAG part)
                        # in a subsequent layer or layer 0. This can be tricky.
                        # For simplicity, let's put them after the max layer found in the DAG part.
                        G.nodes[node]['layer_attr'] = max_layer_val + 1


                pos = nx.multipartite_layout(G, subset_key='layer_attr', orientation='horizontal', scale=2, align='vertical')
                layout_method = "NetworkX multipartite (horizontal)"
                print(f"Layout method: {layout_method}")

            except nx.NetworkXUnfeasible as e_dag: # Graph is not a DAG or other multipartite issue
                print(f"Multipartite layout failed (e.g., graph not a DAG): {e_dag}. Falling back to spring_layout.")
                pos = nx.spring_layout(G, k=1.5/max(1, np.sqrt(G.number_of_nodes())), iterations=50)
                layout_method = "NetworkX spring"
                print(f"Layout method: {layout_method}")
            except Exception as e_multi: # Catch any other multipartite errors
                print(f"Multipartite layout failed with an unexpected error: {e_multi}. Falling back to spring_layout.")
                pos = nx.spring_layout(G, k=1.5/max(1, np.sqrt(G.number_of_nodes())), iterations=50)
                layout_method = "NetworkX spring"
                print(f"Layout method: {layout_method}")
        
        if pos is None: # If somehow pos is still None (e.g. G empty initially but nodes added later)
            if G.number_of_nodes() > 0:
                print("Position calculation failed unexpectedly. Using spring_layout.")
                pos = nx.spring_layout(G, k=1.5/max(1, np.sqrt(G.number_of_nodes())), iterations=50)
                layout_method = "NetworkX spring (fallback)"
            else: # G is truly empty
                print("Graph is empty, cannot compute positions.")

    # Drawing the graph (ensure pos is not None if G has nodes)
    if pos and G.nodes(): # Check if G.nodes() is not empty and pos is computed
        nx.draw(G, pos, labels=node_labels, with_labels=True, node_color=node_colors,
                node_size=3500, font_size=8, font_weight='bold', arrows=True, arrowsize=20, edge_color='darkgray')

        if stage_color_map:
            legend_handles = [plt.Rectangle((0,0),1,1, color=stage_color_map[s_idx]) for s_idx in sorted(stage_color_map.keys())]
            legend_labels = [f"Stage {s_idx}" for s_idx in sorted(stage_color_map.keys())]
            if any(c == 'lightgrey' for c in node_colors):
                 legend_handles.append(plt.Rectangle((0,0),1,1, color='lightgrey'))
                 legend_labels.append("Not in a stage / Unassigned")
            plt.legend(legend_handles, legend_labels, title="Node Stages", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.title(f"{title} (Layout: {layout_method})")
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
    elif G.nodes(): # Nodes exist but pos is None (should not happen with the logic above but as a safe guard)
        print("Warning: Node positions could not be computed. Drawing may be incorrect or fail.")
        # Could attempt a default draw or skip if pos is critical.
        # For now, let it try to proceed which might raise an error if pos is required by draw.
        # nx.draw(G, labels=node_labels, ...) would use a default layout if pos is None.
        nx.draw(G, labels=node_labels, with_labels=True, node_color=node_colors,
                node_size=3500, font_size=8, font_weight='bold', arrows=True, arrowsize=20, edge_color='darkgray')
        plt.title(f"{title} (Layout: Default/Failed)")
        plt.tight_layout()

    # Saving or showing the plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_file_path = os.path.join(output_dir, output_filename)
        plt.savefig(save_file_path, bbox_inches='tight')
        print(f"Computation graph saved to {save_file_path}")
        plt.close(fig)
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"plt.show() failed: {e}. Consider specifying an output_dir to save the plot.")
            plt.close(fig)


def visualize_chip_allocation_per_stage(alloc_info, nodes_in_stage_reassigned_ids,
                                        p_cores, q_cores, stage_idx,
                                        reassigned_id_to_node_id=None,
                                        title_prefix="Chip Allocation",
                                        output_dir=None, output_filename_template="chip_stage_{}.png",
                                        node_cmap_name='Set3'):
    fig = plt.figure(figsize=(max(8, q_cores * 1.0), max(6, p_cores * 1.0)))
    ax = fig.gca() # Get axes for the main plot
    chip_matrix_indices = np.full((p_cores, q_cores), -1)  # Stores indices for the colormap
    core_texts = [[[] for _ in range(q_cores)] for _ in range(p_cores)]

    unique_node_ids_in_stage = sorted(list(set(nodes_in_stage_reassigned_ids)))
    node_id_to_index_map = {node_id: i for i, node_id in enumerate(unique_node_ids_in_stage)}

    for node_local_idx, node_allocs in enumerate(alloc_info):
        reassigned_id = nodes_in_stage_reassigned_ids[node_local_idx]
        original_node_id_str = ""
        if reassigned_id_to_node_id and reassigned_id < len(reassigned_id_to_node_id):
            original_node_id_str = f"(ONNX {reassigned_id_to_node_id[reassigned_id]})"
        
        color_idx_for_map = node_id_to_index_map.get(reassigned_id, -1)

        for replica_idx, replica_allocs in enumerate(node_allocs):
            for core_part_idx, (chip_row, chip_col) in enumerate(replica_allocs):
                if 0 <= chip_row < p_cores and 0 <= chip_col < q_cores:
                    core_id_text_display = chip_row * q_cores + chip_col # Renamed to avoid confusion if 'core_id' has other meaning
                    chip_matrix_indices[chip_row, chip_col] = color_idx_for_map
                    core_texts[chip_row][chip_col].append(
                        # Using the variable for core_id in the f-string
                        f"{core_id_text_display}\nR{reassigned_id}\nRep{replica_idx}-P{core_part_idx}"
                    )
                else:
                    print(f"Warning: Core ({chip_row}, {chip_col}) for R{reassigned_id} out of bounds for chip {p_cores}x{q_cores}.")

    # --- Colormap and Normalization Setup ---
    if unique_node_ids_in_stage:
        node_colors_list = [get_color_for_id(node_id, chosen_cmap_name=node_cmap_name)
                            for node_id in unique_node_ids_in_stage]
    else:
        node_colors_list = [] # Initialize as empty list if no unique nodes

    # final_colors[0] is for empty, final_colors[1:] are for nodes
    final_colors = ['#FFFFFF'] + node_colors_list # White for empty
    # If unique_node_ids_in_stage is empty, node_colors_list is empty, 
    # so final_colors will be ['#FFFFFF']. We need at least two for ListedColormap if using BoundaryNorm.
    if not unique_node_ids_in_stage: # If stage is empty of unique nodes
        final_colors = ['#FFFFFF', '#EEEEEE'] # e.g., White for empty, light grey for "no data" if needed by norm

    final_cmap = matplotlib.colors.ListedColormap(final_colors)
    num_total_colors_for_norm = len(final_colors)


    # Boundaries for the normalization
    boundaries = np.arange(num_total_colors_for_norm + 1) - 0.5
    norm = matplotlib.colors.BoundaryNorm(boundaries, final_cmap.N)

    # Adjust matrix values: empty cells (-1) map to value 0 for cmap
    # Node indices (0 to N-1) map to values 1 to N for cmap
    adjusted_chip_matrix = chip_matrix_indices + 1
    
    mat = ax.matshow(adjusted_chip_matrix, cmap=final_cmap, norm=norm)

    # --- Colorbar Setup ---
    if unique_node_ids_in_stage:
        num_unique_nodes = len(unique_node_ids_in_stage)
        cbar_ticks = np.arange(num_unique_nodes) + 1 

        cbar = fig.colorbar(mat, ax=ax, ticks=cbar_ticks, label="Node ID (color index)",
                            spacing='proportional',
                            boundaries=boundaries, 
                            norm=norm) 
        
        cbar.ax.set_yticklabels([f"R{nid}" for nid in unique_node_ids_in_stage])
    
    # --- Text and Grid ---
    for r in range(p_cores):
        for c in range(q_cores):
            text_content = "\n".join(core_texts[r][c])
            bg_color_index_in_final_cmap = adjusted_chip_matrix[r,c]
            # Ensure index is valid for the cmap, especially if adjusted_chip_matrix could be 0
            # when final_cmap only has one color (e.g. only empty cells, no unique_node_ids_in_stage)
            if bg_color_index_in_final_cmap >= final_cmap.N:
                 bg_color_index_in_final_cmap = final_cmap.N -1 # Cap index if somehow out of bounds for safety

            bg_color_rgba = final_cmap(bg_color_index_in_final_cmap)
            text_color = get_text_color_for_background(bg_color_rgba)
            ax.text(c, r, text_content, va='center', ha='center', fontsize=7, color=text_color)

    ax.set_title(f"{title_prefix} - Stage {stage_idx} ({p_cores}x{q_cores} Chip)")
    ax.set_xlabel("Chip Columns (Q)")
    ax.set_ylabel("Chip Rows (P)")
    ax.set_xticks(np.arange(q_cores))
    ax.set_yticks(np.arange(p_cores))
    ax.set_xticks(np.arange(-.5, q_cores, 1), minor=True)
    ax.set_yticks(np.arange(-.5, p_cores, 1), minor=True)
    ax.grid(which="minor", color="lightgrey", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    fig.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = output_filename_template.format(stage_idx)
        save_file_path = os.path.join(output_dir, filename)
        plt.savefig(save_file_path, bbox_inches='tight')
        print(f"Chip allocation for stage {stage_idx} saved to {save_file_path}")
        plt.close(fig)
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"plt.show() failed for stage {stage_idx}: {e}. Consider specifying an output_dir.")
            plt.close(fig)


# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    PLOTS_OUTPUT_DIRECTORY = "tmp/subtle"

    mock_reassigned_id_graph_nx_friendly = [[1], [2], [], [2]]
    mock_reassigned_id_to_node_id = [100, 101, 102, 103] # R0=ONNX100, etc.
    mock_stages = [
        ([[[ (0,0), (0,1)]]], [2], [[2]], set()), # Stage 0: Node 2
        ([[[ (0,0)]], [[(0,1), (1,1)]]], [1, 3], [[1], [2]], {(1,3)}), # Stage 1: Nodes 1, 3
        ([[[ (0,0), (0,1), (1,0), (1,1)]]], [0], [[4]], set()) # Stage 2: Node 0
    ]
    mock_P_cores = 2
    mock_Q_cores = 2

    print(f"Visualizing mock computation graph with subtle colors, saving to '{PLOTS_OUTPUT_DIRECTORY}'...")
    visualize_computation_graph(
        mock_reassigned_id_graph_nx_friendly,
        mock_stages,
        mock_reassigned_id_to_node_id,
        output_dir=PLOTS_OUTPUT_DIRECTORY,
        output_filename="example_computation_graph.png",
        stage_cmap_name='Pastel1' # Explicitly using Pastel1 for stages
    )

    print(f"\nVisualizing chip allocation with subtle colors, saving to '{PLOTS_OUTPUT_DIRECTORY}'...")
    for i, stage_data in enumerate(mock_stages):
        alloc, nodes_r_ids, _, _ = stage_data
        print(f"Generating plot for Stage {i} with nodes {nodes_r_ids}")
        visualize_chip_allocation_per_stage(
            alloc_info=alloc,
            nodes_in_stage_reassigned_ids=nodes_r_ids,
            p_cores=mock_P_cores,
            q_cores=mock_Q_cores,
            stage_idx=i,
            reassigned_id_to_node_id=mock_reassigned_id_to_node_id,
            output_dir=PLOTS_OUTPUT_DIRECTORY,
            output_filename_template="example_chip_stage_{}.png",
            node_cmap_name='Pastel2' # Using Pastel2 for nodes on chip, or try 'Set3'
        )
    print(f"\nAll example visualizations saved to '{PLOTS_OUTPUT_DIRECTORY}'.")