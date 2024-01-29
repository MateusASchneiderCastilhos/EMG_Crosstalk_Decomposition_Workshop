import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

###### Defining global variables and functions that will be used by the graphs ######
### You do not need understand the code below because it is a configuration for
# the plots of this workshop. You just need to run this cell  ###

# Initilaize global variable
color_scale = np.zeros(2)

# Changing default plot font size
plt.rcParams["font.size"] = 12

# Function to plot the source signals in time domain
def plot_sources_time_domain(y1, y2, sampling_frequency=2048):
    """
    Parameters
    ----------
    y1 (numpy array): array containing the samples of the first source signal

    y2 (numpy array): array containing the samples of the second source signal

    sampling_frequency (float): the sampling frequency (Hz), defaults to 2048 Hz
    """

    # Getting the color scale to plot the join PDF of the signals (seen as
    # random variables) in the plot_sources_and_observations function
    global color_scale
    color_scale = y1

    # Plotting first source signal s_1
    plt.figure()
    plt.plot(np.arange(y1.__len__()) / sampling_frequency, y1, color=(0.9412, 0.3922, 0.3922, 0.8))
    plt.xlabel("Time (s)")
    plt.ylabel("$s_1(t)$")
    plt.show()

    # Plotting second source signal s_2
    plt.figure()
    plt.plot(np.arange(y2.__len__()) / sampling_frequency, y2, color=(0.3922, 2 * 0.3922, 1.0, 0.8))
    plt.xlabel("Time (s)")
    plt.ylabel("$s_2(t)$")
    plt.show()

    # Showing the user the mean and std of the source signals
    print("\n\nMean s1 = ", round(y1.mean(), 3), "\t\tStd s1 = ", round(y1.std(ddof=1), 3)),
    print("Mean s2 = ", round(y2.mean(), 3), "\t\tStd s2 = ", round(y2.std(ddof=1), 3), "\n\n\n")

# Function to plot the joint PDF of the source or observation signals and their histograms
def plot_sources_and_observations(x, y, var="sources", eigen=False, H=np.array([])):
    """
    Parameters
    ----------
    x and y (numpy array): arrays containing the samples of the first and second
    signals to be plotted

    var (string): flag variable that indicates if x and y are source or observation
    signals. Defaults to 'source', if 'observations' or another value, the signals
    are considered observations

    eigen (bool): if computes the eigen vectors of the data and plot them

    H (numpy array): the mixing matrix
    """

    # Plotting the Joint Distribuition of the signals x and y
    fig = plt.figure()
    grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)

    plt.subplots_adjust(0.18, 0.16, 0.95, 0.95)

    main_ax = fig.add_subplot(grid[1:, :-1])
    y_hist = fig.add_subplot(grid[1:, -1], yticklabels=[])
    x_hist = fig.add_subplot(grid[0, :-1], xticklabels=[])

    x_left, x_right = x.min(), x.max()
    y_bottom, y_top = y.min(), y.max()

    main_ax.plot(np.array([x_left, x_right]), np.zeros(2), "k")
    main_ax.plot(np.zeros(2), np.array([y_bottom, y_top]), "k")

    main_ax.scatter(x, y, s=0.5, c=color_scale, cmap="rainbow")

    y_hist.hist(
        y,
        100,
        histtype="bar",
        rwidth=0.8,
        density=True,
        orientation="horizontal",
        color=(0.3922, 2 * 0.3922, 1.0, 0.8),
    )

    x_hist.hist(
        x,
        100,
        histtype="bar",
        rwidth=0.8,
        density=True,
        orientation="vertical",
        color=(0.9412, 0.3922, 0.3922, 0.8),
    )

    # Verifyin the variable types to set the x and y axis labels
    if var.lower() == "observations" or len(var) > 12 or var.lower() != 'sources':

        if len(var) > 12:
            tag = var[12:].replace(" ", "")
            tag = tag.replace("-", "\_")
            tag = tag.replace("_", "\_")
            label = "$x_2^{" + tag + "}$"
            main_ax.set_ylabel(label)
            label = "$x_1^{" + tag + "}$"
            main_ax.set_xlabel(label)
        else:
            main_ax.set_ylabel("$x_2$")
            main_ax.set_xlabel("$x_1$")

        # Plotting the eigenvectors of the covariance matrix os the signals or
        # the vectors that corresponds to the columns of the mixing matrix H
        if eigen or H.shape[0] > 0:

            if H.shape[0] > 0:
                U = H
                quiver_label1 = "$\\bf{h}_{1}$"
                quiver_label2 = "$\\bf{h}_{2}$"

            else:
                Cxx = np.cov(np.array([x - np.mean(x), y - np.mean(y)]))
                d, U = np.linalg.eig(Cxx)
                quiver_label1 = "$\\bf{u}_{1}$"
                quiver_label2 = "$\\bf{u}_{2}$"

            origin = np.array([0, 0])

            scale = max(abs(x_left), abs(x_right))
            scale = max(scale, abs(y_bottom))
            scale = max(scale, abs(y_top)) / 3
            scale = max(abs(U[:, 0])) / scale

            u1 = main_ax.quiver(*origin, *(U[:, 0]), color="r", scale_units="xy", scale=scale)
            u2 = main_ax.quiver(*origin, *(U[:, 1]), color="b", scale_units="xy", scale=scale)

            main_ax.quiverkey(
                u1,
                U[0, 0] / scale,
                U[1, 0] / scale,
                1,
                quiver_label1,
                labelpos="E",
                coordinates="data",
                visible=False,
            )
            main_ax.quiverkey(
                u2,
                U[0, 1] / scale,
                U[1, 1] / scale,
                1,
                quiver_label2,
                labelpos="E",
                coordinates="data",
                visible=False,
            )

    else:
        main_ax.set_ylabel("$s_2$")
        main_ax.set_xlabel("$s_1$")

    # Defining the axis limits and labels
    y_hist.set_ylim(main_ax.get_ylim())
    x_labels = y_hist.get_xticks()
    x_labels = x_labels[:-1]
    y_hist.set_xticks(x_labels)
    y_hist.set_xticklabels(x_labels, rotation=30)
    x_hist.set_xlim(main_ax.get_xlim())

    plt.show()

# Function to import the matlab file that contains the EMG data
def import_Mat_file(filePath: str):

    if filePath[-4:] != ".mat":

        filePath = "".join([filePath, ".mat"])

    matFile = scio.loadmat(filePath)

    f_sampling = matFile["SamplingFrequency"][0, 0]

    time_samples = matFile["Time"][0, 0].reshape(-1)

    sEMG = matFile["Data"][0, 0]

    if sEMG.shape[1] < sEMG.shape[0]:
        sEMG = sEMG.T

    return f_sampling, time_samples, sEMG


###### Defining contrast functions that can be used in the FastICA algorithm ######

class Skewness:
    """Class that defines the Skweness function, G(w) = (w^3)/3, and its first and second derivatives.
    Then it can be use as Cost Function in the Fixed Point Algorithm (fastICA).
    Function used in (NEGRO, et al., 2016) and (HOLOBAR, 2007).
    """

    @staticmethod
    def g(w: np.ndarray) -> np.ndarray:
        """First derivative of Skewness function
        G(w) = (w^3)/3 -> dG(w)/dw = g(w) = w^2
        Parameters
        ----------
            w ([float]): The i-th separation vector.
        Returns
        ----------
            ([float]): The i-th separation vector with all elments being the squared of the original.
        """
        return np.square(w)

    @staticmethod
    def dg_dw(w: np.ndarray) -> np.ndarray:
        """Second derivative of Skewness function
        G(w) = (w^3)/3 -> d^2G(w)/dw^2 = dg(w)/dw = 2*w
        Parameters
        ----------
            w ([float]): The i-th separation vector.
        Returns
        ----------
            ([float]): The i-th separation vector with all elments being the twice the original.
        """

        return 2 * w

class Kurtosis:
    """Class that defines the Kurtosis function, G(w) = (w^4)/4, and its first and second derivatives.
    Then it can be use as Cost Function in the Fixed Point Algorithm (fastICA).
    Function suggested in (HYVÄRINEN, 1999).
    """

    @staticmethod
    def g(w: np.ndarray) -> np.ndarray:
        """First derivative of Kurtosis function
        G(w) = (w^4)/4 -> dG(w)/dw = g(w) = w^3
        Parameters
        ----------
            w ([float]): The i-th separation vector.
        Returns
        ----------
            ([float]): The i-th separation vector with all elments being the squared of the original.
        """
        return np.power(w, 3)

    @staticmethod
    def dg_dw(w: np.ndarray) -> np.ndarray:
        """Second derivative of Kurtosis function
        G(w) = (w^4)/4 -> d^2G(w)/dw^2 = dg(w)/dw = 3*w^2
        Parameters
        ----------
            w ([float]): The i-th separation vector.
        Returns
        ----------
            ([float]): The i-th separation vector with all elments being the twice the original.
        """

        return 3 * np.square(w)


class LogCosh:
    """Class that defines the Log Cosh function, G(w) = log(cosh(w)), and its first and second derivatives.
    Then it can be use as Cost Function in the Fixed Point Algorithm (fastICA). Here is used the natural
    logarithm, so the cost function becomes G(w) = ln(cosh(w)).
    Function suggested in (HYVÄRINEN, 1999).
    """

    @staticmethod
    def g(w: np.ndarray) -> np.ndarray:
        """First derivative of Log Cosh function
        G(w) = ln(cosh(w)) -> dG(w)/dw = g(w) = tanh(w)
        Parameters
        ----------
            w ([float]): The i-th separation vector.
        Returns
        ----------
            ([float]): The i-th separation vector with all elments being the hyperbolic tangent of the original elements.
        """
        # If is desired use logarithms in another base, by example, in base 10, then the return must be:
        # return np.tanh(w) / np.log(10)
        return np.tanh(w)

    @staticmethod
    def dg_dw(w: np.ndarray) -> np.ndarray:
        """Second derivative of Log Cosh function
        G(w) = ln(cosh(w)) -> d^2G(w)/dw^2 = dg(w)/dw = sech^2(w) = 1 - tanh^2(w)
        Parameters
        ----------
            w ([float]): The i-th separation vector.
        Returns
        ----------
            ([float]): The i-th separation vector with all elments being the hyperbolic squared secant of the original elements.
        """
        # If is desired use logarithms in another base, by example, in base 10, then the return must be:
        # return (1 - np.square(np.tanh(w))) / np.log(10)
        return (1 - np.square(np.tanh(w)))


class ExpSquared:
    """Class that defines the Exponetial of w squared function, G(w) = exp(-(w^2)/2), and its first and
    second derivatives. Then it can be use as Cost Function in the Fixed Point Algorithm (fastICA).
    Function suggested in (HYVÄRINEN, 1999).
    """

    @staticmethod
    def g(w: np.ndarray) -> np.ndarray:
        """First derivative of Exponetial of w squared function
        G(w) = exp(-(w^2)/2) -> dG(w)/dw = g(w) = -exp(-(w^2)/2)*w
        Parameters
        ----------
            w ([float]): The i-th separation vector.
        Returns
        ----------
            ([float]): The i-th separation vector with all elments being the G(w) function
            times the negative of the original elements.
        """

        return (-w) * np.exp((-1 / 2) * w * w)

    @staticmethod
    def dg_dw(w: np.ndarray) -> np.ndarray:
        """Second derivative of Exponetial of w squared function
        G(w) = exp(-(w^2)/2) -> d^2G(w)/dw^2 = dg(w)/dw = (exp(-(w^2)/2))*(w^2 - 1)
        Parameters
        ----------
            w ([float]): The i-th separation vector.
        Returns
        ----------
            ([float]): The i-th separation vector with all elments being the G(w) function
            times the squared original elements minus 1.
        """

        return ((w * w) - 1) * np.exp((-1 / 2) * w * w)

###### Defining the FastICA algorithm ######

def fastICA(z: np.ndarray, M: int = 120, max_iter: int = 50, Tolx: float = 0.0001, cost: int = 1):
    """
    FastICA algorithm proposed by (Hyvärinen, Oja, 1997) and (Hyvärinen, 1999) to estimate the
    projecction vector w. This algorithm is a fixed point algorithm with orthogonalization and
    normalization steps, for a better estimation.

    Parameters
    ----------
        z (numpy array): Whitened extended observation matrix.
        M (int): Number of iterations of the whole algorithm (that is, the maximum number of possibly estimated sources).
        max_iter (int): Maximum number of iterations that fastICA will run to find an estimated
            separation vector on the i-th iteration of main FOR loop.
        Tolx (float): The toleration or convergence criteria that sepration vectors from fastICA must
            satisfy.

    Returns
    ----------
        B (numpy array): Separation matrix, which each column is an array correspondig
        to current estimation of the projection vector.

    """

    B: np.ndarray = np.zeros((z.shape[0], M), dtype=float)
    BB: np.ndarray = 0 * np.identity(z.shape[0])

    if cost == 2:
        cf = Kurtosis
    elif cost == 4:
        cf = ExpSquared
    elif cost == 3:
        cf = LogCosh
    else:
        cf = Skewness

    for i in range(M):

        """
        1. Initialize the vector w_i(0) and w_i(-1) with unit norm
        """
        w_new = np.random.rand(z.shape[0])
        vec_norm = np.linalg.norm(w_new)
        if vec_norm > 0:
            w_new /= vec_norm

        """
                2. While |w_i(n)^{T}w_i(n - 1) - 1| > (0.0001 = Tolx)
            """
        n = 0
        while True and n < max_iter:

            w_old = np.copy(w_new)

            """
                    a. Fixed point algorithm
                        w_i(n) = E{zg[w_i(n - 1)^{T}z]} - Aw_i(n - 1)
                        with A = E{g'[w_i(n - 1)^{T}z]}
                """
            s = np.dot(w_old, z)
            w_new = (z * cf.g(s)).mean(axis=1) - cf.dg_dw(s).mean() * w_old

            """
                    b. Orthogonalization
                        w_i(n) = w_i(n) - BB^{T}w_i(n)
                """
            w_new -= np.dot(BB, w_new)

            """
                    c. Normalization
                        w_i(n) = w_i(n)/||w_i(n)||
                """
            vec_norm = np.linalg.norm(w_new)
            if vec_norm > 0:
                w_new /= vec_norm

            # Recalculate convergence criterion
            tolx = np.absolute(np.dot(w_new, w_old) - 1)

            if tolx <= Tolx:
                break

            """
                    d. Set n = n + 1
                """
            n += 1

        B[:, i] = w_new
        BB += np.dot(w_new.reshape(-1, 1), w_new.reshape(1, -1))

    return B


###### Defining a code to identify and remove repeated/duplicate source signals ######
### The methdology below is out off the scope of this workshop, if you want to
#   know more about it, please contact mateus.aschneider@gmail.com using the subject
#   Decomposition Worshop - Finding Duplicates MUs                              ###
def finding_duplicates(
    extracted_PTs: np.ndarray,
    sil: np.ndarray,
    cov: np.ndarray,
    R: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identify and remove the array indexe of repeated (or duplicated) MU spike trains.

    Returns the indexes of non-repeated MU spike trains and the pairs of MU numbers
    that might be repeated MUs.

    Parameters
    ----------
    extracted_PTs : ndarray
        Two dimensional binary (0 or 1) array with the rows indicating the i-th spike train and
        the columns the j-th time sample. The samples equal to one indicate a spike.

    sil : ndarray
        One dimensional array containing the SIL values of the correspondent spike trains, i.e.,
        the i-th SIL value corresponds to the i-th spike train.

    cov : ndarray
        One dimensional array containing the CoV of the interspike intervals for each spike trains.
        The i-th CoV value corresponds to the i-th spike train.
    
    R : int
        The extension factor value used to created delayed versions of the EMG signals.
    
    Returns
    -------
    out1 : ndarray
        An 1-D array of the indexes that corresponds to the non-repeated MUs. Then, doing 
        extracted_PTs[out1] the non-repeated spike trains are obtained, as well as for sil and cov.

    out2 : ndarray
        An 2-D array where each row is a pair or MUs that might be the same MU (repeated). The array values
        are the MU numbers. If out2 - 1 is done it is obtained an array with the indexes of possible
        repeated MUs, these indexes are contained in out1.

    """

    alpha: int = 2 * R
    RoCD: float = 0.5
    duration: int = extracted_PTs.shape[1]
    itera: int
    comp_win: int
    step: int
    interval_beg: float
    interval_end: float
    mean_interval: float
    n_MUs: int
    i: int
    j: int
    count: int
    a_j: int
    b_j: int
    c_j: int
    indexes_to_analyse: np.ndarray = np.array([], dtype=int)
    indexes: np.ndarray = np.array([], dtype=int)
    pt: np.ndarray = np.array([], dtype=int)
    mini_cst: np.ndarray = np.zeros((alpha - 1, duration), dtype=int)
    pt_aux: np.ndarray = np.zeros((2 * alpha + 1, duration), dtype=int)
    not_MU_indexes: np.ndarray = np.array([], dtype=int)
    possible_dup_MU: np.ndarray = np.zeros((1, 2), dtype=int)
    n_spikes: np.ndarray = np.count_nonzero(extracted_PTs, axis=1)
    mu_indexes: np.ndarray = np.argsort(n_spikes, kind="mergesort")

    print("\nIdentifying and removing duplicated Pulse Trains...\n")

    mu_indexes = np.delete(mu_indexes, np.argwhere(cov[mu_indexes] == -1).reshape(-1))

    interval_beg = mu_indexes.__len__() * 0.125
    interval_beg = int(interval_beg) if (interval_beg - int(interval_beg) < 0.5) else (int(interval_beg) + 1)
    interval_end = mu_indexes.__len__() * 0.875
    interval_end = int(interval_end) if (interval_end - int(interval_end) < 0.5) else (int(interval_end) + 1)
    mean_interval = n_spikes[mu_indexes[interval_beg:interval_end]].mean()

    mu_indexes = np.delete(mu_indexes, np.argwhere(n_spikes[mu_indexes] <= 0.05 * mean_interval).reshape(-1))

    n_spikes: int = mu_indexes.__len__()

    print("\n\tPulse Trains considered outliers (removed): ", extracted_PTs.__len__() - n_spikes, "\n")

    comp_win = int(n_spikes * 0.25)
    if comp_win < 3:
        comp_win = n_spikes

    step = int(comp_win * 0.5)

    mu_indexes = mu_indexes[::-1]

    for itera in range(int(n_spikes / step + 1) - 1):

        interval_beg = itera * step
        interval_end = interval_beg + comp_win

        if interval_end > n_spikes:
            interval_end = n_spikes
            interval_beg = interval_end - int(1.5 * step) - 1

        indexes_to_analyse = mu_indexes[interval_beg:interval_end]

        pt = extracted_PTs[indexes_to_analyse]

        n_MUs = indexes_to_analyse.__len__()

        for i in range(n_MUs):

            if indexes_to_analyse[i] not in not_MU_indexes:

                a_j = np.count_nonzero(pt[i, :])

                pt_aux *= 0
                mini_cst *= 0

                for count in range(alpha):
                    count_plus = count + 1
                    count_minus = (-1) * count_plus
                    pt_aux[count, count_plus:] = pt[i, :count_minus]
                    pt_aux[count_minus - 1, :count_minus] = pt[i, count_plus:]

                pt_aux[-1, :] = pt[i, :]

                count_plus = 2
                for count in range(0, R - 1):
                    count_plus += 2

                    mini_cst[count, :] = pt_aux[2 * count : count_plus, :].sum(axis=0)
                    mini_cst[-count - 2, :] = pt_aux[-count_plus - 1 : -2 * count - 1, :].sum(axis=0)

                mini_cst[-1, :] = pt_aux[[0, 1, -1, -2, -3], :].sum(axis=0)

                indexes = np.argwhere(mini_cst > 1)

                if indexes.__len__() > 0:
                    mini_cst[indexes[:, 0], indexes[:, 1]] = 1

                for j in range(i + 1, n_MUs):

                    b_j = np.count_nonzero(pt[j, :])

                    c_j = np.count_nonzero((mini_cst + pt[j, :]) == 2, axis=1).max()

                    if (c_j / a_j) >= RoCD and (c_j / b_j) >= RoCD:

                        if sil[indexes_to_analyse[j]] > sil[indexes_to_analyse[i]]:

                            not_MU_indexes = np.append(not_MU_indexes, indexes_to_analyse[i])

                        elif sil[indexes_to_analyse[j]] == sil[indexes_to_analyse[i]]:

                            if cov[indexes_to_analyse[j]] < cov[indexes_to_analyse[i]]:

                                not_MU_indexes = np.append(not_MU_indexes, indexes_to_analyse[i])

                            else:

                                not_MU_indexes = np.append(not_MU_indexes, indexes_to_analyse[j])

                        else:

                            not_MU_indexes = np.append(not_MU_indexes, indexes_to_analyse[j])

                not_MU_indexes = np.unique(not_MU_indexes)

    mu_indexes = mu_indexes[np.isin(mu_indexes, not_MU_indexes, invert=True)]

    not_MU_indexes = np.array([], dtype=int)

    extracted_PTs = extracted_PTs[mu_indexes]
    sil = sil[mu_indexes]
    cov = cov[mu_indexes]
    n_MUs = mu_indexes.__len__()

    for i in range(n_MUs):

        if i not in not_MU_indexes:

            a_j = np.count_nonzero(extracted_PTs[i, :])

            pt_aux *= 0
            mini_cst *= 0

            for count in range(alpha):
                count_plus = count + 1
                count_minus = (-1) * count_plus
                pt_aux[count, count_plus:] = extracted_PTs[i, :count_minus]
                pt_aux[count_minus - 1, :count_minus] = extracted_PTs[i, count_plus:]

            pt_aux[-1, :] = extracted_PTs[i, :]

            count_plus = 2
            for count in range(0, R - 1):
                count_plus += 2

                mini_cst[count, :] = pt_aux[2 * count : count_plus, :].sum(axis=0)
                mini_cst[-count - 2, :] = pt_aux[-count_plus - 1 : -2 * count - 1, :].sum(axis=0)

            mini_cst[-1, :] = pt_aux[[0, 1, -1, -2, -3], :].sum(axis=0)

            indexes = np.argwhere(mini_cst > 1)

            if indexes.__len__() > 0:
                mini_cst[indexes[:, 0], indexes[:, 1]] = 1

            for j in range(i + 1, n_MUs):

                b_j = np.count_nonzero(extracted_PTs[j, :])

                c_j = np.count_nonzero((mini_cst + extracted_PTs[j, :]) == 2, axis=1).max()

                if (c_j / b_j) >= RoCD:

                    if (c_j / a_j) >= RoCD:

                        if sil[j] > sil[i]:

                            not_MU_indexes = np.append(not_MU_indexes, i)

                        elif sil[j] == sil[i]:

                            if cov[j] < cov[i]:

                                not_MU_indexes = np.append(not_MU_indexes, i)

                            else:

                                not_MU_indexes = np.append(not_MU_indexes, j)

                        else:

                            not_MU_indexes = np.append(not_MU_indexes, j)

                    else:

                        possible_dup_MU = np.append(possible_dup_MU, mu_indexes[[i, j]].reshape(1, -1), axis=0)

            not_MU_indexes = np.unique(not_MU_indexes)

    mu_indexes = np.delete(mu_indexes, not_MU_indexes)

    extracted_PTs = np.delete(extracted_PTs, not_MU_indexes, axis=0)

    possible_dup_MU = np.delete(possible_dup_MU, 0, axis=0)
    possible_dup_MU = np.delete(
        possible_dup_MU, np.argwhere(np.isin(possible_dup_MU, mu_indexes, invert=True))[:, 0], axis=0
    )

    if possible_dup_MU.__len__() > 0:
        j = 0
        for i in possible_dup_MU:
            possible_dup_MU[j, :] = np.argwhere(np.isin(mu_indexes, i)).reshape(-1)
            j += 1

        possible_dup_MU = 1 + np.sort(possible_dup_MU, axis=1)

    print("\n\tPulse Trains considered duplicated (removed): ", n_spikes - mu_indexes.__len__(), "\n")
    print("\n\tUnique Pulse Trains: ", mu_indexes.__len__(), "\n")
    print("\n\tPossible Remained Duplicated MUs: ", possible_dup_MU, "\n")

    return mu_indexes, possible_dup_MU