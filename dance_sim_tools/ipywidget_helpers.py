#Custom function for easier construction of ipywidget plots.

import ipywidgets as widgets


def slider(start,stop,step,init):#,init):
    return widgets.FloatSlider(
    value=init,
    min=start,
    max=stop,
    step=step,
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.2f',
)
