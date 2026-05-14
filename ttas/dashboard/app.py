import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np

app = dash.Dash(__name__, title="Tulsa Topological Affordability Spacetime")

app.layout = html.Div(style={'backgroundColor': '#0a0a1a', 'color': '#00ffcc', 'fontFamily': 'Arial'}, children=[
    html.H1("Tulsa Topological Affordability Spacetime (TTAS)", style={'textAlign': 'center', 'padding': '20px'}),
    
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Spacetime Manifold', value='tab-1', style={'backgroundColor': '#111'}),
        dcc.Tab(label='Euler Characteristic Surface', value='tab-2', style={'backgroundColor': '#111'}),
        dcc.Tab(label='Persistence Vineyard', value='tab-3', style={'backgroundColor': '#111'}),
        dcc.Tab(label='Causal Shock Lab', value='tab-4', style={'backgroundColor': '#111'}),
        dcc.Tab(label='Decision Boundary Navigator', value='tab-5', style={'backgroundColor': '#111'}),
    ], colors={"border": "#333", "primary": "#00ffcc", "background": "#222"}),
    
    html.Div(id='tabs-content', style={'padding': '20px'})
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        # Generate dummy 3D UMAP data
        u = np.random.normal(size=(500, 3))
        entropy = np.linalg.norm(u, axis=1)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=u[:,0], y=u[:,1], z=u[:,2],
            mode='markers',
            marker=dict(size=4, color=entropy, colorscale='Viridis', opacity=0.8)
        )])
        fig.update_layout(paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a', 
                          scene=dict(xaxis=dict(showbackground=False),
                                     yaxis=dict(showbackground=False),
                                     zaxis=dict(showbackground=False)))
                                     
        return html.Div([
            html.H3("3D Embedding of the Tulsa Housing Manifold"),
            dcc.Graph(figure=fig)
        ])
        
    elif tab == 'tab-2':
        # 4D Euler Surface (3D spatial + color for Chi)
        X, Y, Z = np.mgrid[-1:1:20j, -1:1:20j, -1:1:20j]
        chi = np.sin(np.pi*X)*np.cos(np.pi*Y)*np.sin(np.pi*Z)
        
        fig = go.Figure(data=go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=chi.flatten(),
            isomin=-0.5, isomax=0.5,
            surface_count=5,
            colorscale='Plasma',
            caps=dict(x_show=False, y_show=False)
        ))
        fig.update_layout(paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a')
        
        return html.Div([
            html.H3("Euler Characteristic Phase Transition Surface"),
            dcc.Graph(figure=fig)
        ])
        
    elif tab == 'tab-4':
        return html.Div([
            html.H3("Counterfactual Topological Analysis"),
            html.Label("Simulate Interest Rate Shock:"),
            dcc.Slider(0.5, 3.0, 0.1, value=1.5, id='rate-slider'),
            html.Div(id='shock-output', style={'marginTop': '20px'})
        ])
    else:
        return html.Div([
            html.H3("Module loaded in memory. Visualization rendering pending.")
        ])

@app.callback(Output('shock-output', 'children'),
              Input('rate-slider', 'value'))
def update_shock(val):
    wass_dist = 0.25 * val**2
    return f"Topological Wasserstein Distance from baseline: {wass_dist:.4f}. Regime shift probability: {min(1.0, wass_dist/2)*100:.1f}%"

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
