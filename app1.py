from flask import Flask, request, render_template
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            n = int(request.form['number_of_individuals'])
            matrix_scores = request.form['matrix_scores'].strip().split('\n')
            phq_9_scores = request.form['phq_9_scores'].strip().split(',')

            Matrix_score = [list(map(float, row.split(','))) for row in matrix_scores]
            PHQ_9 = list(map(float, phq_9_scores))

            G = nx.DiGraph()

            for i in range(n):
                G.add_node(i + 1)
                for j in range(n):
                    if i != j:
                        weight = Matrix_score[i][j]
                        if not np.isnan(weight):
                            G.add_edge(i + 1, j + 1, weight=weight)

            pos = nx.spring_layout(G, dim=3)
            pos_3d = {node: (pos[node][0], pos[node][1], PHQ_9[node-1]) for node in G.nodes()}

            weights = [G[u][v]['weight'] for u, v in G.edges()]
            max_weight = max(weights)
            min_weight = min(weights)
            normalized_weights = [(weight - min_weight) / (max_weight - min_weight) for weight in weights]

            fig = go.Figure()

            # Add edges with colors mapped to the 'Portland' colorscale
            for edge, weight in zip(G.edges(data=True), normalized_weights):
                x0, y0, z0 = pos_3d[edge[0]]
                x1, y1, z1 = pos_3d[edge[1]]

                fig.add_trace(go.Scatter3d(
                    x=[x0, x1], y=[y0, y1], z=[z0, z1], mode='lines',
                    line=dict(color='rgba(0,0,255,' + str(weight) + ')', width=10),
                    hoverinfo='none'
                ))

            # Add nodes
            node_x, node_y, node_z = [], [], []
            for node in G.nodes():
                x, y, z = pos_3d[node]
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)

            node_trace = go.Scatter3d(
                x=node_x, y=node_y, z=node_z, mode='markers+text',
                marker=dict(size=10, color='skyblue', line=dict(color='black', width=2)),
                text=[str(node) for node in G.nodes()], textposition="top center"
            )
            fig.add_trace(node_trace)

            # Dummy trace for colorbar
            colorbar_trace = go.Scatter3d(
                x=[None], y=[None], z=[None], mode='markers',
                marker=dict(
                    size=10, 
                    color=normalized_weights, 
                    colorscale='Portland', 
                    cmin=min_weight, 
                    cmax=max_weight, 
                    showscale=True,
                    colorbar=dict(title='Edge Weight')
                ),
                hoverinfo='none'
            )
            fig.add_trace(colorbar_trace)

            fig.update_layout(
                title='3D Network Graph with PHQ-9 Score',
                titlefont_size=16,
                showlegend=False,
                scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='PHQ-9 Score'))
            )

            graph_html = fig.to_html(full_html=False)
            return graph_html
        except Exception as e:
            return f"An error occurred: {str(e)}"

    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
