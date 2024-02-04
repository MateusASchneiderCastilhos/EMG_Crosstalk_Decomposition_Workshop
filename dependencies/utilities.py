import ipywidgets as ipywi
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

# ****************** GENERAL FUNCTIONS - BOTH NOTEBOOKS ******************

def config_plots():
    """
    Configures the default plot settings for improved aesthetics.

    This function updates the Matplotlib parameters to customize plot appearance.
    It removes the top and right spines from the axes and sets the font size to 12.

    Parameters
    ----------
    - None

    Returns
    ----------
    - None
    """

    plt.rcParams.update(
        {
            "axes.spines.top" : False,
            "axes.spines.right" : False,
            'font.size': 12,
        }
    )

def import_Mat_file(filePath: str):
    """
    Imports High-Density (HD) sEMG signals saved as a '.mat' file.

    Parameters
    ----------
    - filePath (str): The path to the '.mat' file with the filename with/without the extension.

    Returns
    ----------
    - f_sampling (float): Sampling frequency of the sEMG signals.
    - time_samples (numpy.ndarray): Array containing the time samples of the sEMG data.
    - sEMG (numpy.ndarray): Matrix containing the HD sEMG data. Rows represent EMG channels, and
                            columns represent the sampled data.

    Notes
    ----------
    - The '.mat' file must be saved as a dictionary with specific keys:
        - 'SamplingFrequency': Array containing the sampling frequency.
        - 'Time': 2D array containing the time samples.
        - 'Data': 2D array containing the HD sEMG data.

    - The function reshapes the data matrix if needed to ensure consistent channel and time orientation.
    """

    if filePath[-4:] != ".mat":
        filePath = "".join([filePath, ".mat"])

    matFile = scio.loadmat(filePath)

    f_sampling = matFile["SamplingFrequency"][0, 0]
    time_samples = matFile["Time"][0, 0].reshape(-1)
    sEMG = matFile["Data"][0, 0]

    if sEMG.shape[1] != time_samples.shape[0]:
        sEMG = sEMG.T

    return f_sampling, time_samples, sEMG

#################### Contrast Functions For MG_Crosstalk_Decomposition_Workshop and Decomposition_Panel ####################
class Squared_CF:
    """Class that defines the first and second derivatives of the equation G(w) = x^3/3.
    Then it can be use as a Cost Function in the Fixed Point Algorithm (fastICA).
    """

    @staticmethod
    def g(w: np.ndarray):
        """First derivative of G(w) = (w^3)/3 -> dG(w)/dw = g(w) = w^2.

        Parameters
        ----------
        - w (np.ndarray): The i-th estimated source.

        Returns
        ----------
        - (np.ndarray): The values of g(w) applied to the i-th estimated source.
        """
        return np.square(w)

    @staticmethod
    def dg_dw(w: np.ndarray) :
        """Second derivative of G(w) = (w^3)/3 -> d^2G(w)/dw^2 = dg(w)/dw = 2*w.

        Parameters
        ----------
        - w (np.ndarray): The i-th estimated source.
        Returns

        ----------
        - (np.ndarray): The values of dg(w)/dw applied to the i-th estimated source.
        """
        return 2 * w

class Log_CF:
    """Class that defines the first and second derivatives of the equation G(x) = x log(x^2 + 1) + 2 (arctan(x) - x) where log() is the natural logarithm.
    Then it can be use as a Cost Function in the Fixed Point Algorithm (fastICA).
    """

    @staticmethod
    def g(w: np.ndarray):
        """First derivative of G(x) = x log(x^2 + 1) + 2 (arctan(x) - x) -> dG(w)/dw = g(w) = log(w^2 + 1).

        Parameters
        ----------
        - w (np.ndarray): The i-th estimated source.

        Returns
        ----------
        - (np.ndarray): The values of g(w) applied to the i-th estimated source.
        """
        # If a different base for logarithms is desired, for example, base 10, the return must be:
        # return np.log10(w * w + 1)
        return np.log(w * w + 1)

    @staticmethod
    def dg_dw(w: np.ndarray):
        """Second derivative of G(x) = x log(x^2 + 1) + 2 (arctan(x) - x) -> d^2G(w)/dw^2 = dg(w)/dw = 2 * w / (w * w + 1).

        Parameters
        ----------
        - w (np.ndarray): The i-th separation vector.
        
        Returns
        ----------
        - (np.ndarray): The values of dg(w)/dw applied to the i-th estimated source.
        """
        # If a different base for logarithms is desired, for example, base 10, the return must be:
        # return 2 / (w * w + 1) / np.log10(10)
        return 2 * w / (w * w + 1)

class Skewness_CF:
    """Class that defines the first and second derivatives of the equation G(w) = (w^4)/4.
    Then it can be use as a Cost Function in the Fixed Point Algorithm (fastICA).
    """

    @staticmethod
    def g(w: np.ndarray):
        """First derivative of G(w) = (w^4)/4 -> dG(w)/dw = g(w) = w^3.

        Parameters
        ----------
        - w (np.ndarray): The i-th separation vector.

        Returns
        ----------
        - (np.ndarray): The values of g(w) applied to the i-th estimated source.
        """
        return np.power(w, 3)

    @staticmethod
    def dg_dw(w: np.ndarray) :
        """Second derivative of G(w) = (w^4)/4 -> d^2G(w)/dw^2 = dg(w)/dw = 3*w^2.

        Parameters
        ----------
        - w (np.ndarray): The i-th separation vector.
        
        Returns
        ----------
        - (np.ndarray): The values of dg(w)/dw applied to the i-th estimated source.
        """

        return 3 * np.square(w)

class LogCosh_CF:
    """Class that defines the first and second derivatives of the equation G(w) = log(cosh(w)) where log() is the natural logarithm.
    Then it can be use as a Cost Function in the Fixed Point Algorithm (fastICA).
    """

    @staticmethod
    def g(w: np.ndarray):
        """First derivative of  G(w) = ln(cosh(w)) -> dG(w)/dw = g(w) = tanh(w).

        Parameters
        ----------
        - w (np.ndarray): The i-th separation vector.

        Returns
        ----------
        - (np.ndarray): The values of g(w) applied to the i-th estimated source.
        """
        # If is desired use logarithms in another base, by example, in base 10, then the return must be:
        # return np.tanh(w) / np.log(10)
        return np.tanh(w)

    @staticmethod
    def dg_dw(w: np.ndarray):
        """Second derivative of G(w) = ln(cosh(w)) -> d^2G(w)/dw^2 = dg(w)/dw = sech^2(w) = 1 - tanh^2(w).

        Parameters
        ----------
        - w (np.ndarray): The i-th separation vector.

        Returns
        ----------
        - (np.ndarray): The values of dg(w)/dw applied to the i-th estimated source.
        """
        # If is desired use logarithms in another base, by example, in base 10, then the return must be:
        # return (1 - np.square(np.tanh(w))) / np.log(10)
        return (1 - np.square(np.tanh(w)))

class ExpSquared_CF:
    """Class that defines the first and second derivatives of the equation G(w) = exp(-(w^2)/2).
    Then it can be use as a Cost Function in the Fixed Point Algorithm (fastICA).
    """

    @staticmethod
    def g(w: np.ndarray):
        """First derivative of G(w) = exp(-(w^2)/2) -> dG(w)/dw = g(w) = -exp(-(w^2)/2)*w.

        Parameters
        ----------
        - w (np.ndarray): The i-th separation vector.

        Returns
        ----------
        - (np.ndarray): The values of g(w) applied to the i-th estimated source.
        """

        return (-w) * np.exp((-1 / 2) * w * w)

    @staticmethod
    def dg_dw(w: np.ndarray):
        """Second derivative of G(w) = exp(-(w^2)/2) -> d^2G(w)/dw^2 = dg(w)/dw = (exp(-(w^2)/2))*(w^2 - 1).

        Parameters
        ----------
        - w (np.ndarray): The i-th separation vector.

        Returns
        ----------
        - (np.ndarray): The values of dg(w)/dw applied to the i-th estimated source.
        """

        return ((w * w) - 1) * np.exp((-1 / 2) * w * w)
