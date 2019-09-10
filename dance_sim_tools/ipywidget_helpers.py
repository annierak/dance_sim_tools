#Custom function for easier construction of ipywidget plots.

import ipywidgets as widgets


def slider(name,start,stop,step,init,readout_format='.0f'):#,init):
    return widgets.FloatSlider(
    value=init,
    min=start,
    max=stop,
    step=step,
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format=readout_format,
    description=name
)
