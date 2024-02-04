import matplotlib.pyplot as plt
import numpy as np

from dependencies.utilities import Squared_CF, Skewness_CF, LogCosh_CF, ExpSquared_CF


# Function to plot the source signals in time domain
def plot_sources_time_domain(y1, y2, title1, title2, sampling_frequency=2048):
    """
    Plot the two source signals in the time domain.

    Parameters
    ----------
    - y1 (numpy.ndarray): array containing the samples of the first source signal

    - y2 (numpy.ndarray): array containing the samples of the second source signal

    - sampling_frequency (float): the sampling frequency (Hz), defaults to 2048 Hz

    Returns
    ----------
    - None
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
    if title1 != '' :
        plt.title(title1) 
    plt.show()

    # Plotting second source signal s_2
    plt.figure()
    plt.plot(np.arange(y2.__len__()) / sampling_frequency, y2, color=(0.3922, 2 * 0.3922, 1.0, 0.8))
    plt.xlabel("Time (s)")
    plt.ylabel("$s_2(t)$")
    if title2 != '' :
        plt.title(title2)
    plt.show()

    # Showing the user the mean and std of the source signals
    print("\n\n\t Mean and Standard Deviation - Original Source Signals")
    print("\tMean s1 = ", round(y1.mean(), 3), "\t\tStd s1 = ", round(y1.std(ddof=1), 3)),
    print("\tMean s2 = ", round(y2.mean(), 3), "\t\tStd s2 = ", round(y2.std(ddof=1), 3), "\n\n\n")

# Function to plot the joint PDF of the source or observation signals and their histograms
def plot_sources_and_observations(x, y, title, var="sources", eigen=False, H=np.array([])):
    """
    Plot joint PDF of two signals and their histograms.

    Parameters
    ----------
    - x and y (numpy.ndarray): Arrays containing the samples of the first and second signals to be plotted.
    - var (string): Flag variable indicating if x and y are source or observation signals. Defaults to 'sources';
                    if 'observations' or another value, the signals are considered observations.
    - eigen (bool): If True, compute the eigen vectors of the data and plot them.
    - H (numpy.ndarray): The mixing matrix.

    Returns
    ----------
    - None
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

        if var == "estimated sources":
            main_ax.set_ylabel("$\hat{s}_2$")
            main_ax.set_xlabel("$\hat{s}_1$")
        elif len(var) > 12:
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

    # Defining plot title
    if title != '':
        plt.title(title)

    # Defining the axis limits and labels
    y_hist.set_ylim(main_ax.get_ylim())
    x_labels = y_hist.get_xticks()
    x_labels = x_labels[:-1]
    y_hist.set_xticks(x_labels)
    y_hist.set_xticklabels(x_labels, rotation=30)
    x_hist.set_xlim(main_ax.get_xlim())

    plt.show()

###### Defining the FastICA algorithm ######
def fastICA(z: np.ndarray, M: int = 120, max_iter: int = 50, Tolx: float = 0.0001, cost: int = 1):
    """
    FastICA algorithm proposed by (Hyvärinen, Oja, 1997) and (Hyvärinen, 1999) to estimate the
    projecction vector w. This algorithm is a fixed point algorithm with orthogonalization and
    normalization steps, for a better estimation.

    Parameters
    ----------
    - z (numpy.ndarray): Whitened extended observation matrix.
    - M (int): Number of iterations of the whole algorithm (that is, the maximum number of possibly estimated sources).
    - max_iter (int): Maximum number of iterations that fastICA will run to find an estimated
                      separation vector on the i-th iteration of main FOR loop.
    - Tolx (float): The toleration or convergence criteria that sepration vectors from fastICA must
                    satisfy.

    Returns
    ----------
    - B (numpy.ndarray): Separation matrix, which each column is an array correspondig
                         to current estimation of the projection vector.

    """

    B: np.ndarray = np.zeros((z.shape[0], M), dtype=float)
    BB: np.ndarray = 0 * np.identity(z.shape[0])

    if cost == 2:
        cf = Skewness_CF
    elif cost == 4:
        cf = ExpSquared_CF
    elif cost == 3:
        cf = LogCosh_CF
    else:
        cf = Squared_CF

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
