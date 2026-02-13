#!/usr/bin/env python3
"""
Visualize stable score computation to verify correctness.

This script shows:
1. Object mesh
2. Center of gravity (COG) 
3. Sample grasp points with gripper plane normals
4. Color-coded by stable score (green=low/stable, red=high/unstable)

Usage:
    python experiments/visualize_stable_score.py --graspnet_root /datasets/graspnet --obj_id 0
"""

import os
import sys
import argparse
import numpy as np

# Add parent to path for utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trimesh
import plotly.graph_objects as go

from utils.stable_score_utils import (
    compute_mesh_cog, 
    generate_grasp_views, 
    compute_grasp_plane_normal
)


def load_object_data(graspnet_root, obj_id):
    """Load mesh and grasp labels for an object."""
    obj_id_str = str(obj_id).zfill(3)
    
    # Mesh path
    mesh_path = os.path.join(graspnet_root, 'models', obj_id_str, 'nontextured.ply')
    
    # Grasp label path
    label_path = os.path.join(graspnet_root, 'grasp_label', f'{obj_id_str}_labels.npz')
    
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Labels not found: {label_path}")
    
    mesh = trimesh.load(mesh_path)
    labels = np.load(label_path)
    
    return mesh, labels, mesh_path


def compute_stable_scores_subset(grasp_points, cog, views, sample_points=20, sample_views=10):
    """
    Compute stable scores for a subset of points and views for visualization.
    """
    Np = min(sample_points, grasp_points.shape[0])
    V = min(sample_views, views.shape[0])
    
    # Sample points evenly
    point_indices = np.linspace(0, grasp_points.shape[0]-1, Np, dtype=int)
    view_indices = np.linspace(0, views.shape[0]-1, V, dtype=int)
    
    results = []
    
    for p_idx in point_indices:
        grasp_point = grasp_points[p_idx]
        
        for v_idx in view_indices:
            view = views[v_idx]
            plane_normal = compute_grasp_plane_normal(view)
            
            # Distance from COG to plane
            cog_to_point = cog - grasp_point
            distance = np.abs(np.dot(plane_normal, cog_to_point))
            
            results.append({
                'point': grasp_point,
                'view': view,
                'normal': plane_normal,
                'distance': distance,
                'p_idx': p_idx,
                'v_idx': v_idx
            })
    
    # Normalize distances
    max_dist = max(r['distance'] for r in results)
    if max_dist > 1e-8:
        for r in results:
            r['stable_score'] = r['distance'] / max_dist
    else:
        for r in results:
            r['stable_score'] = 0.0
    
    return results


def create_visualization(mesh, cog, results, show_planes=True, arrow_scale=0.03):
    """Create plotly visualization of stable scores."""
    
    fig = go.Figure()
    
    # Add mesh
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightblue',
        opacity=0.5,
        name='Object Mesh'
    ))
    
    # Add COG as large marker
    fig.add_trace(go.Scatter3d(
        x=[cog[0]],
        y=[cog[1]],
        z=[cog[2]],
        mode='markers',
        marker=dict(size=12, color='magenta', symbol='diamond'),
        name='Center of Gravity (COG)'
    ))
    
    # Colorblind-friendly gradient: Viridis-style (dark purple -> blue -> teal -> yellow)
    # score 0 = dark purple (stable), score 1 = bright yellow (unstable)
    def score_to_color(score):
        # Viridis-inspired colorblind-friendly gradient
        # Interpolate through: purple (0) -> blue (0.25) -> teal (0.5) -> green (0.75) -> yellow (1)
        if score < 0.25:
            t = score / 0.25
            r = int(68 + t * (59 - 68))
            g = int(1 + t * (82 - 1))
            b = int(84 + t * (139 - 84))
        elif score < 0.5:
            t = (score - 0.25) / 0.25
            r = int(59 + t * (33 - 59))
            g = int(82 + t * (145 - 82))
            b = int(139 + t * (140 - 139))
        elif score < 0.75:
            t = (score - 0.5) / 0.25
            r = int(33 + t * (94 - 33))
            g = int(145 + t * (201 - 145))
            b = int(140 + t * (98 - 140))
        else:
            t = (score - 0.75) / 0.25
            r = int(94 + t * (253 - 94))
            g = int(201 + t * (231 - 201))
            b = int(98 + t * (37 - 98))
        return f'rgb({r},{g},{b})'
    
    # Add grasp points colored by stable score
    for r in results:
        point = r['point']
        stable_score = r['stable_score']
        color = score_to_color(stable_score)
        
        fig.add_trace(go.Scatter3d(
            x=[point[0]],
            y=[point[1]],
            z=[point[2]],
            mode='markers',
            marker=dict(size=10, color=color, line=dict(width=1, color='white')),
            name=f"Point {r['p_idx']}, View {r['v_idx']}, Score: {stable_score:.3f}",
            showlegend=False,
            hovertemplate=(
                f"<b>Stable Score: {stable_score:.3f}</b><br>"
                f"Point Index: {r['p_idx']}<br>"
                f"View Index: {r['v_idx']}<br>"
                f"Raw Distance: {r['distance']:.4f}<br>"
                f"Position: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})"
                f"<extra></extra>"
            )
        ))
        
        # Draw arrow showing plane normal (approach direction)
        normal = r['normal']
        end = point + normal * arrow_scale
        
        fig.add_trace(go.Scatter3d(
            x=[point[0], end[0]],
            y=[point[1], end[1]],
            z=[point[2], end[2]],
            mode='lines',
            line=dict(color=color, width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Draw dashed line from COG to grasp point to visualize distance
        if stable_score < 0.3:  # Only for stable grasps
            fig.add_trace(go.Scatter3d(
                x=[cog[0], point[0]],
                y=[cog[1], point[1]],
                z=[cog[2], point[2]],
                mode='lines',
                line=dict(color='rgba(68,1,84,0.4)', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add a dummy trace for colorbar
    scores = [r['stable_score'] for r in results]
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(
            size=0.1,
            color=[0, 1],
            colorscale='Viridis',
            colorbar=dict(
                title='Stable Score',
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=['0 (Stable)', '0.25', '0.5', '0.75', '1 (Unstable)'],
                x=1.02
            ),
            showscale=True
        ),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Layout
    fig.update_layout(
        title=f'Stable Score Visualization<br><sup>Purple = Stable (COG close to plane), Yellow = Unstable (COG far from plane)</sup>',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_single_grasp_detail(mesh, cog, result, arrow_scale=0.05):
    """
    Create detailed visualization of a single grasp showing:
    - Mesh
    - COG
    - Grasp point
    - Gripper plane (as a small disc)
    - Perpendicular distance line
    """
    fig = go.Figure()
    
    # Mesh
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightblue',
        opacity=0.3,
        name='Object'
    ))
    
    point = result['point']
    normal = result['normal']
    
    # COG
    fig.add_trace(go.Scatter3d(
        x=[cog[0]], y=[cog[1]], z=[cog[2]],
        mode='markers+text',
        marker=dict(size=12, color='red', symbol='diamond'),
        text=['COG'],
        textposition='top center',
        name='COG'
    ))
    
    # Grasp point
    fig.add_trace(go.Scatter3d(
        x=[point[0]], y=[point[1]], z=[point[2]],
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=['Grasp Point'],
        textposition='bottom center',
        name='Grasp Point'
    ))
    
    # Approach direction arrow
    end = point + normal * arrow_scale
    fig.add_trace(go.Scatter3d(
        x=[point[0], end[0]],
        y=[point[1], end[1]],
        z=[point[2], end[2]],
        mode='lines+text',
        line=dict(color='purple', width=5),
        text=['', 'Plane Normal'],
        textposition='top center',
        name='Plane Normal (Approach Dir)'
    ))
    
    # Project COG onto gripper plane
    cog_to_point = cog - point
    perp_distance = np.dot(normal, cog_to_point)
    cog_projected = cog - perp_distance * normal
    
    # Line from COG to its projection (perpendicular distance)
    fig.add_trace(go.Scatter3d(
        x=[cog[0], cog_projected[0]],
        y=[cog[1], cog_projected[1]],
        z=[cog[2], cog_projected[2]],
        mode='lines+text',
        line=dict(color='green', width=4, dash='dash'),
        text=[f'd = {abs(perp_distance):.4f}', ''],
        textposition='middle center',
        name=f'Perp. Distance = {abs(perp_distance):.4f}'
    ))
    
    # Draw gripper plane as a disc
    # Create circle in local coordinate frame
    if abs(normal[0]) < 0.9:
        up = np.array([1, 0, 0])
    else:
        up = np.array([0, 1, 0])
    
    right = np.cross(normal, up)
    right /= np.linalg.norm(right)
    up_orth = np.cross(right, normal)
    
    plane_radius = arrow_scale * 1.5
    theta = np.linspace(0, 2*np.pi, 20)
    circle_points = []
    for t in theta:
        p = point + plane_radius * (np.cos(t) * right + np.sin(t) * up_orth)
        circle_points.append(p)
    circle_points = np.array(circle_points)
    
    fig.add_trace(go.Scatter3d(
        x=circle_points[:, 0],
        y=circle_points[:, 1],
        z=circle_points[:, 2],
        mode='lines',
        line=dict(color='purple', width=3),
        name='Gripper Plane (edge)'
    ))
    
    # Mark projected point
    fig.add_trace(go.Scatter3d(
        x=[cog_projected[0]], y=[cog_projected[1]], z=[cog_projected[2]],
        mode='markers+text',
        marker=dict(size=8, color='orange'),
        text=['COG Projection'],
        textposition='bottom center',
        name='COG Projected'
    ))
    
    fig.update_layout(
        title=f'Grasp Detail - Stable Score: {result["stable_score"]:.3f}<br>'
              f'<sup>Distance from COG to gripper plane: {abs(perp_distance):.4f}</sup>',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y', 
            zaxis_title='Z',
            aspectmode='data'
        )
    )
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize stable score computation')
    parser.add_argument('--graspnet_root', type=str, default='/datasets/graspnet',
                        help='Path to GraspNet dataset root')
    parser.add_argument('--obj_id', type=int, default=0,
                        help='Object ID to visualize (0-87)')
    parser.add_argument('--num_points', type=int, default=20,
                        help='Number of grasp points to sample')
    parser.add_argument('--num_views', type=int, default=5,
                        help='Number of views per point to sample')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML file (default: stable_score_viz_objXXX.html)')
    parser.add_argument('--show_detail', action='store_true',
                        help='Also show detailed view of best and worst grasps')
    
    args = parser.parse_args()
    
    print(f"Loading object {args.obj_id}...")
    mesh, labels, mesh_path = load_object_data(args.graspnet_root, args.obj_id)
    
    grasp_points = labels['points']
    print(f"Object has {grasp_points.shape[0]} grasp points")
    
    print("Computing COG...")
    cog = compute_mesh_cog(mesh_path)
    print(f"COG: {cog}")
    
    print("Generating views...")
    views = generate_grasp_views(300)
    
    print(f"Computing stable scores for {args.num_points} points × {args.num_views} views...")
    results = compute_stable_scores_subset(
        grasp_points, cog, views, 
        sample_points=args.num_points, 
        sample_views=args.num_views
    )
    
    # Statistics
    scores = [r['stable_score'] for r in results]
    print(f"\nStable score statistics:")
    print(f"  Min: {min(scores):.4f}")
    print(f"  Max: {max(scores):.4f}")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Std: {np.std(scores):.4f}")
    
    # Find best and worst
    best = min(results, key=lambda r: r['stable_score'])
    worst = max(results, key=lambda r: r['stable_score'])
    
    print(f"\nMost stable grasp: Point {best['p_idx']}, View {best['v_idx']}, Score {best['stable_score']:.4f}")
    print(f"Least stable grasp: Point {worst['p_idx']}, View {worst['v_idx']}, Score {worst['stable_score']:.4f}")
    
    # Create main visualization
    print("\nCreating visualization...")
    fig = create_visualization(mesh, cog, results)
    
    output_file = args.output or f"stable_score_viz_obj{str(args.obj_id).zfill(3)}.html"
    fig.write_html(output_file)
    print(f"Saved visualization to: {output_file}")
    
    if args.show_detail:
        # Create detail views
        fig_best = create_single_grasp_detail(mesh, cog, best)
        fig_best.write_html(f"stable_score_best_obj{str(args.obj_id).zfill(3)}.html")
        print(f"Saved best grasp detail")
        
        fig_worst = create_single_grasp_detail(mesh, cog, worst)
        fig_worst.write_html(f"stable_score_worst_obj{str(args.obj_id).zfill(3)}.html")
        print(f"Saved worst grasp detail")
    
    print("\nDone! Open the HTML file in a browser to interact with the visualization.")


if __name__ == '__main__':
    main()
