import ipywidgets as widgets
import numpy as np
from ipywidgets import GridspecLayout, Layout
from IPython.display import display, Latex, HTML, Markdown
from backend.print_functions import *
from backend.matrix import Matrix
from backend.primary_decomposition import projections_to_latex

gs = None


def create_grid(n_features):
    global gs
    n = n_features
    gs = GridspecLayout(n_features, n_features, layout=Layout(
        width=f'{n_features*90}px', height='auto'))
    for i in range(n_features):
        for j in range(n_features):
            gs[i, j] = widgets.FloatText(
                layout=Layout(width='80px', height='auto'))
    return gs


n_features_slider = widgets.IntSlider(
    value=5,
    min=2,
    max=10,
    step=1,
    description='Matrix Dimension:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d',
    layout={'width': '350px'},
    style={'description_width': 'initial'}
)

out = widgets.Output()


def display_result(b):
    # load matrix
    gs_values = []
    for i in range(n_features_slider.value):
        gs_values_row = []
        for j in range(n_features_slider.value):
            gs_values_row.append(gs[i, j].value)
        gs_values.append(gs_values_row)
    gs_array = np.asarray(gs_values)
    m = Matrix(gs_array)
    # clear output
    out.clear_output()
    # redraw output
    d = 'not' if not m.isDiagonal else ''
    with out:
        display(Latex(f'characteristic polynomial:{get_packed_polynomial_latex(m.charPoly)}'),
                Latex(
                    f'minimal polynomial:{get_packed_polynomial_latex(m.minPoly)}'),
                Latex(f'A is {d} diagonalizable'),
                Latex(f'Jordan normal form is: $J = {print_matrix(m.J)}$'),
                Latex(f'P mejardenet is $  {print_matrix(m.P)}$'),
                Latex(f'Jordan–Chevalley decomposition:'),
                Latex(f'$$S = {print_matrix(m.S)}$$'),
                Latex(f'$$N = {print_matrix(m.N)}$$'),
                *projections_to_latex(m)
        )

    b.description = 'recalculate'
    return


def start():
    display(HTML(r'<style> body { direction: ltr; } </style>'))
    out.clear_output()
    widgets.interact(create_grid, n_features=n_features_slider)
    button = widgets.Button(description="calculate")
    display(button)
    display(out)
    button.on_click(display_result)
