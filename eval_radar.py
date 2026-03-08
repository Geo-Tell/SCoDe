import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def draw_radar_plot(data_dict, labels, title, filename=None, use_fills=False):
    """
    Draw a radar chart comparing multiple sets of data with BASE values as the starting point.
    
    Args:
        data_dict: Dictionary with keys as model names and values as data arrays
        labels: Labels for the radar chart axes
        title: Title for the chart (will be used for filename but not displayed)
        filename: Optional filename for saving the chart
        use_fills: Whether to fill the radar areas
    """
    # Create radar chart
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Base color mapping for methods (without resolution consideration)
    base_color_map = {
        'BASE': '#888888',       # Gray for BASE
        'OETR': '#1f77b4',       # Blue for OETR
        'SCoDe': '#ff7f0e',      # Orange for SCoDe
    }
    
    # Extract all resolutions from model keys to determine transparency levels
    resolutions = set()
    methods = set()
    for model_key in data_dict.keys():
        if model_key != 'BASE' and '-' in model_key:
            method, res = model_key.split('-', 1)
            try:
                res_num = int(res)
                resolutions.add(res_num)
                methods.add(method)
            except ValueError:
                pass
    
    # Sort resolutions to assign transparency consistently
    sorted_resolutions = sorted(resolutions) if resolutions else []
    
    # Create transparency mapping (distribute evenly across resolution types)
    def get_alpha(resolution):
        if not sorted_resolutions or len(sorted_resolutions) == 1:
            return 1.0
        
        # Find the index of this resolution in the sorted list
        try:
            res_index = sorted_resolutions.index(resolution)
        except ValueError:
            return 1.0
        
        # Distribute alpha values evenly: higher resolution gets higher alpha
        num_resolutions = len(sorted_resolutions)
        # Map index to alpha: 0.4 to 1.0, evenly distributed
        alpha = 0.4 + 0.6 * res_index / (num_resolutions - 1)
        print(f"Resolution {resolution} (index {res_index}/{num_resolutions-1}) mapped to alpha {alpha:.2f}")
        return alpha
    
    # Get BASE values to use as reference points
    base_values = data_dict.get('BASE', None)
    
    if base_values is None:
        print("Warning: BASE values not found. Using absolute values.")
        relative_mode = False
    else:
        relative_mode = True
    
    # Data for plotting
    plot_data = {}
    
    if relative_mode:
        # Normalize data relative to BASE
        for model_name, data in data_dict.items():
            if model_name == 'BASE':
                # BASE model is represented as zeros (no improvement)
                plot_data[model_name] = np.zeros_like(data)
            else:
                # Calculate percentage improvements over BASE
                plot_data[model_name] = [(data[i] - base_values[i]) for i in range(len(data))]
    else:
        # Use absolute values
        plot_data = data_dict
    
    # Find min and max values for better y-axis limits
    all_values = []
    for model, values in plot_data.items():
        if model != 'BASE' or not relative_mode:  # Skip BASE in relative mode as it's all zeros
            all_values.extend(values)
    
    min_value = min(all_values) if all_values else 0
    max_value = max(all_values) if all_values else 100
    
    # Add padding to limits
    padding = (max_value - min_value) * 0.1
    ax.set_ylim(min_value - padding, max_value + padding)
    
    # Plot each dataset
    for model_name, data in plot_data.items():
        data_closed = np.concatenate((data, [data[0]]))  # Close the loop
        angles_closed = angles + angles[:1]
        
        if model_name == 'BASE' and relative_mode:
            # Draw BASE as a reference circle at zero, without a label
            ax.plot(angles_closed, data_closed, 
                    linewidth=2, linestyle='-', color=base_color_map['BASE']) 
        else:
            # For other models, extract method and resolution
            if '-' in model_name:
                method, res_str = model_name.split('-', 1)
                try:
                    resolution = int(res_str)
                    alpha = get_alpha(resolution)
                except ValueError:
                    method = model_name
                    alpha = 1.0
            else:
                method = model_name
                alpha = 1.0
            
            # Get base color for the method
            base_color = base_color_map.get(method, '#1f77b4')  # Default to blue if method not found
            
            # Plot with method-consistent color and resolution-based transparency
            ax.plot(angles_closed, data_closed, linewidth=2.5, color=base_color, 
                   alpha=alpha, label=model_name)
            
            # Optional fill with low alpha
            if use_fills:
                ax.fill(angles_closed, data_closed, color=base_color, alpha=alpha*0.15)
    
    # Set chart properties
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=16, fontweight='bold') # Increased fontsize from 10 to 14
    ax.tick_params(axis='x', pad=15) # Add padding to move labels further out
    
    # Format y-ticks
    y_ticks = ax.get_yticks()
    if relative_mode:
        # Add "+" sign for positive improvements
        ax.set_yticklabels([f"+{y:.1f}%" if y > 0 else f"{y:.1f}%" for y in y_ticks], fontsize=16) # Increased fontsize from 12 to 14
    else:
        ax.set_yticklabels([f'{int(y)}%' for y in y_ticks], fontsize=16) # Increased fontsize from 12 to 14
    
    ax.grid(True, linestyle='-', alpha=0.3)
    
    # Remove note about BASE values (now hidden)
    
    # Add legend with improved styling but no title
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=16, framealpha=0.7) # Increased fontsize from 12 to 14
    
    # Hide title (but keep it for filename use)
    # plt.title(title, size=18, y=1.1, fontweight='bold')
    
    # Create output directory if it doesn't exist
    os.makedirs('outputs/radar_eval', exist_ok=True)
    
    # Save the radar chart
    if filename is None:
        # Generate a filename based on the title if none provided
        filename = title.replace(' ', '_').lower() + '.png'
    
    save_path = os.path.join('outputs/radar_eval', filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Radar chart saved to {save_path}")


def process_csv_and_create_plots(csv_path):
    """
    Read the CSV file and create radar plots for each method
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Print column names to verify structure
    print("CSV columns:", df.columns.tolist())
    print("Sample data rows:")
    print(df.head())
    
    # Get unique methods
    methods = df['Methods'].unique()
    print("Unique methods found:", methods)
    
    # Metrics to include in the radar plot, with newlines before '@'
    metrics_raw = ['AUC@5', 'AUC@10', 'AUC@20', 'Acc@5', 'Acc@10', 'Acc@15', 'Acc@20', 
                   'mAP@5', 'mAP@10', 'mAP@20', 'P', 'MS']
    metrics_display = [m.replace('@', '\n@') if '@' in m else m for m in metrics_raw]
    
    # Process each method separately
    for method in methods:
        method_df = df[df['Methods'] == method]
        print(f"\nProcessing method: {method}")
        print(f"Found {len(method_df)} entries for this method")
        
        # List unique models for this method
        unique_models = method_df['Model'].unique()
        unique_sizes = method_df['Size'].unique() if 'Size' in df.columns else []
        print(f"Models found: {unique_models}")
        print(f"Sizes found: {unique_sizes}")
        
        # Create a dictionary to store data for each model
        data_dict = {}
        
        # Extract BASE values
        base_rows = method_df[method_df['Model'] == 'BASE']
        if len(base_rows) > 0:
            base_row = base_rows.iloc[0]
            base_values = []
            for metric in metrics_raw: # Use raw metrics for data lookup
                if metric in base_row:
                    base_values.append(float(base_row[metric]))
                else:
                    print(f"Warning: Metric {metric} not found for BASE model")
            
            if len(base_values) == len(metrics_raw): # Check against raw metrics count
                data_dict['BASE'] = base_values
                print(f"Added BASE values: {base_values}")
        else:
            print("No BASE model found for this method")
        
        # Extract values for other models
        for _, row in method_df.iterrows():
            model = row['Model']
            if model == 'BASE':
                continue
                
            # Get the size (if available)
            size = row['Size'] if 'Size' in row else 'unknown'
            
            # Create model key
            model_key = f"{model}-{size}"
            
            # Extract values for metrics
            model_values = []
            valid_metrics = True
            for metric in metrics_raw: # Use raw metrics for data lookup
                if metric in row:
                    model_values.append(float(row[metric]))
                else:
                    print(f"Warning: Metric {metric} not found for {model_key}")
                    valid_metrics = False
            
            if valid_metrics:
                data_dict[model_key] = model_values
                print(f"Added {model_key} values: {model_values}")
        
        print(f"Final data dictionary keys: {list(data_dict.keys())}")
        
        # Skip if we don't have enough data
        if len(data_dict) <= 1:
            print(f"Skipping {method} - not enough model data")
            continue
        
        # Create radar plot for this method
        draw_radar_plot(
            data_dict,
            metrics_display, # Pass display labels with newlines to the plot function
            f'{method} Performance Comparison',
            filename=f'{method}_radar_comparison.png',
            use_fills=False
        )

def draw_vertical_bars():
    """
    Draw tall, narrow vertical bar chart comparing model runtimes
    """
    # Data in milliseconds
    # runtimes = [108.585, 115.635, 114.305]  # SCoDe runtimes at 640 and 480 resolutions
    runtimes = [108.13, 100.12, 91.39]  # SCoDe runtimes at 640 and 480 resolutions
    
    # Use full resolution format for x-axis labels
    resolutions = ["1024×1024", "640×640", "480×480"]
    resolution_values = [1024, 640, 480]  # For alpha calculation
    
    # Use the same color scheme as radar chart
    base_color = '#ff7f0e'  # Orange for SCoDe
    
    # Apply the same alpha calculation logic as radar chart
    sorted_resolutions = sorted(resolution_values)
    
    # Create figure and axis with taller, narrower dimensions
    fig, ax = plt.subplots(figsize=(2.0, 8))
    
    # Adjust subplot position to give more space for y-axis label
    plt.subplots_adjust(left=0.25)
    
    # Plot vertical bars individually with matching colors and alphas
    bars = []
    for i, (runtime, res) in enumerate(zip(runtimes, resolution_values)):
        res_index = sorted_resolutions.index(res)
        num_resolutions = len(sorted_resolutions)
        # Map index to alpha: 0.4 to 1.0, evenly distributed
        alpha = 0.4 + 0.6 * res_index / (num_resolutions - 1)
        
        bar = ax.bar(i, runtime, color=base_color, alpha=alpha, width=0.7)
        bars.extend(bar)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, 
                f'{height:.1f}', ha='center', va='bottom', fontsize=12)
    
    # Set x-axis ticks and labels
    ax.set_xticks(range(len(resolutions)))
    ax.set_xticklabels(resolutions, fontsize=10, rotation=90)
    
    # Set y-axis label with a bit more padding
    ax.set_ylabel('Runtime (ms)', fontsize=16, labelpad=8)
    
    # Remove spines/borders
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    
    # Add light grid lines only
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='both', length=0)
    
    # Adjust y-axis limits for better proportions
    ax.set_ylim(0, max(runtimes) * 1.1)
    
    # Save figure
    os.makedirs('outputs/radar_eval', exist_ok=True)
    save_path = os.path.join('outputs/radar_eval', 'scode_runtime_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Runtime comparison chart saved to {save_path}")


if __name__ == "__main__":
    csv_path = '/data/nfs/lhj/OverlapEstimation/SCoDe/metrics_sizes_mr4.csv'
    # Print file exists status
    print(f"CSV file exists: {os.path.exists(csv_path)}")
    
    process_csv_and_create_plots(csv_path)
    draw_vertical_bars()