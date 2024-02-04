from IPython.display import display

import ipywidgets as ipywi

class decompWidgets():

    checked_channels = {}

    def channels_selection(self, grid):
        """
        This function provides interactive widgets (checkboxes) displayed in a spatial location
        corresponding to the surface grid used during the recording of HD sEMG signals.
        Additionally, this function is responsible for keeping the list of selected (checkbox
        checked) or unselected channels updated. Only the selected channels are used in the
        decomposition processing.

        Parameters
        ----------
        - grid (numpy.ndarray): The electrode grid array.

        Returns
        ----------
        - checked_channels (dict): A dictionary mapping channel numbers to their respective checkboxes.
        """

        my_matrix = []

        num_rows, num_cols = grid.shape

        for row in range(num_rows):
            aux = []
            for col in range(num_cols):
                ch = grid[row, col]
                if ch != 0:
                    check = ipywi.Checkbox(value=True, description=str(grid[row, col]), indent=False)
                    self.checked_channels['ch' + str(ch)] = check
                    aux.append(check)
                else:
                    # Since the electrode grid used to record HD sEMG has 64 electrodes distributed in a matrix
                    # with 13 rows and 5 columns, one element of the matrix is not an electrode. This element
                    # is the 13th row and the 1st column, which here represents the channel number 0.
                    check = ipywi.Checkbox(value=False, description='', indent=False, disabled=True)
                    check.layout.visibility = 'hidden'
                    aux.append(check)

            my_matrix.append(ipywi.HBox(aux, layout=ipywi.Layout(width='25%')))

        display(ipywi.VBox(my_matrix))
        
        return self.checked_channels
    
    def decomposition_menu(self, decomp):
        """
        This function provides five text input widgets (only numbers allowed) where you
        can define the values of the FastICA parameters. There are upper and lower bounds
        specified for each widget, and these bounds are presented before the text input.
        Additionally, there are two selection widgets with several options for parameters,
        but you can choose only one. Lastly, there is a progress bar widget that shows
        "how much time" has passed after starting the decomposition.

        Parameters
        ----------
        - None

        Returns
        ----------
        - List of ipywidgets representing the interactive parameters:
            - BoundedIntText: Number of Iterations (M) [1 - 640]
            - BoundedIntText: Extension Factor (R) [4 - 32]
            - BoundedIntText: Internal Loop Iterations [20 - 60]
            - BoundedFloatText: FastICA Convergence [10ˉ⁵ - 10ˉ²]
            - BoundedFloatText: SIL Threshold (SIL [%]) [0 - 100]
            - RadioButtons: Contrast Function g(x)
            - RadioButtons: Initializations of wⱼ
        """

        style = {'description_width': 'initial'}
        
        w_M = ipywi.BoundedIntText(min = 1, max = 640, step = 1, value = 120, description='Number of Iterations (M) [1 - 640]: ',style=style)
        w_R = ipywi.BoundedIntText(min = 4, max = 32, step = 1, value = 10, description='Extension Factor (R) [4 - 32]: ',style=style)
        w_iterations = ipywi.BoundedIntText(min = 20, max = 60, step = 1, value = 30, description='Internal Loop Iterations [20 - 60]: ',style=style)
        w_Tolx = ipywi.BoundedFloatText(min = 0.00001, max = 0.01, step = 0.000001, value = 0.0023, description='FastICA Convergence [10ˉ⁵ - 10ˉ²]: ',style=style,readout_format='.6f')
        w_Sil = ipywi.BoundedFloatText(min = 0, max = 100, step = 0.1, value = 90, description='SIL Threshold (SIL [%]) [0 - 100]: ',style=style, readout_format='.1f')

        w_cf = ipywi.RadioButtons(
            description='Contrast Function g(x):',
            options=[
                'x²',
                'x³',
                '(-x) exp(-x² / 2)',
                'log(x² + 1)',
                'tanh(x)',
            ],
            layout={'width': 'max-content'},
        )

        w_winit = ipywi.RadioButtons(
            description='Initializations of wⱼ:',
            options=[
                'Maximum',
                'Median',
                '81th Percentile',
                'Random',
            ],
            layout={'width': 'max-content'},
        )

        ipywi.interact_manual(
            decomp.decompose,
            m=w_M,
            r=w_R,
            maxIter=w_iterations,
            tolx=w_Tolx,
            sil=w_Sil,
            contrast=w_cf,
            wInit=w_winit,
            channels = ipywi.fixed(self.checked_channels)
            ).widget.children[7].description = 'Decompose'
        
        
 