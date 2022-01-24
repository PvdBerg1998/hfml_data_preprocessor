from tkinter import *
from tkinter import filedialog
import os
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from scipy.optimize import curve_fit


#
# Utils
#


def gaussian(x, a, mu, sigma2):
    """
    Have a guess what this does
    """
    return a * 1.0 / np.sqrt(2.0*np.pi*sigma2) * np.exp(-(x-mu)**2.0/(2.0*sigma2))


def weight(x, f):
    """
    Weighing factor used to assign a higher priority to data near the peak
    f increases the weight in the middle
    """
    return -(f*x)**4-1


def import_csv(path):
    """
    Reads path as a csv file into a pandas df with "x" and "y" columns
    """
    return pd.read_csv(path, names=["x", "y"])


def fit_peak(data, mu0, var0, dx, f, N):
    """
    Iterative peak finder
    data: pandas df with "x" and "y" columns
    mu0: peak location guess
    var0: variance guess
    dx: x range around peak to include in fit
    f: same as weight() f
    N: iteration count
    """

    # Initial guess
    # amplitude, mu, variance
    p0 = (data[data.x.between(mu0-dx, mu0+dx)].y.max(), mu0, var0)

    # Iterate to move middle of range towards the solution
    for i in range(N):
        subset = data[data.x.between(p0[1]-dx, p0[1]+dx)]
        pfit, _ = curve_fit(
            gaussian,
            xdata=subset.x,
            ydata=subset.y,
            sigma=weight(np.linspace(-1, 1, num=len(subset.x)), f),
            p0=p0,
            bounds=([0, p0[1]-dx, 0], [np.inf, p0[1]+dx, np.inf])
        )
        print(pfit)
        p0 = (p0[0], pfit[1], p0[2])

    return pfit


def fit_path(path, mu0, var0, dx, f, N):
    """
    Import, fit and plot path.
    Parameters are the same as in fit_peak.
    """

    global plot
    plot.clear()

    df = import_csv(path)

    fit = fit_peak(
        df,
        mu0,
        var0,
        dx,
        f,
        N
    )

    mu = fit[1]
    sigma = np.sqrt(fit[2])
    strength = gaussian(mu, fit[0], fit[1], fit[2])

    subset = df[df.x.between(mu0-dx, mu0+dx)]
    subset.plot(ax=plot, x="x", y="y", label="data")
    plot.plot(subset.x, gaussian(
        subset.x, fit[0], fit[1], fit[2]), label="fit")

    out = f"{fit[1]} +- {sigma}. Strength: {strength}"
    print(f"Peak: {out}")
    peak_var.set(out)

    global canvas
    canvas.draw()


def plot_path(path):
    global plot
    plot.clear()

    df = import_csv(path)
    df.plot(ax=plot, x="x", y="y", label="data")

    global canvas
    canvas.draw()


#
# Callbacks
#


def set_defaults():
    global mu0_var
    global var0_var
    global dx_var
    global f_var

    mu0_var.set(200)
    var0_var.set(5)
    dx_var.set(30)
    f_var.set(10)

    global plot
    plot.clear()
    global canvas
    canvas.draw()

    do_plot()


def do_plot():
    global filename_var

    if not os.path.exists(filename_var.get()):
        return

    plot_path(filename_var.get())


def do_fit(*args):
    global filename_var
    global mu0_var
    global var0_var
    global dx_var
    global f_var

    if not os.path.exists(filename_var.get()):
        return

    fit_path(
        path=filename_var.get(),
        mu0=mu0_var.get(),
        var0=var0_var.get(),
        dx=dx_var.get(),
        f=f_var.get(),
        N=5
    )


def path_button_callback():
    global filename_var
    path = filedialog.askopenfilename(filetypes=(("CSV", "*.csv"),))
    print(f"Selected path: {path}")
    filename_var.set(path)
    do_plot()


#
# Entry
#


if __name__ == "__main__":
    # Create TK window
    root = Tk()
    root.title("Peak Extraction")
    root.geometry("1280x1024")
    root.bind('<Return>', do_fit)

    # The figure that will contain the plot
    figure = Figure(figsize=(16, 9), dpi=75)
    global plot
    plot = figure.add_subplot(111)
    plot.autoscale()

    # Place figure on window
    frame = Frame(root)
    frame.pack(fill=BOTH, pady=5)
    global canvas
    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas.get_tk_widget().pack()

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack()

    # Control buttons
    frame = Frame(root)
    frame.pack(fill=X, padx=5, pady=5)
    plot_button = Button(frame, text="Fit",
                         command=do_fit, padx=5, pady=5)
    plot_button.pack(side=LEFT, fill=X, expand=True)
    reset_button = Button(frame, text="Reset",
                          command=set_defaults, padx=5, pady=5)
    reset_button.pack(side=LEFT, padx=5, fill=X, expand=True)
    path_button = Button(frame,
                         text="Open CSV", command=path_button_callback, padx=5, pady=5)
    path_button.pack(side=LEFT, fill=X, expand=True)

    # Filename
    frame = Frame(root)
    frame.pack(fill=X)
    filename_label = Label(frame, text="Path:", padx=5,
                           pady=5, width=20, justify=RIGHT, anchor="e")
    filename_label.pack(side=LEFT)
    global filename_var
    filename_var = StringVar()
    filename_entry = Entry(frame, textvariable=filename_var, state="readonly")
    filename_entry.pack(fill=X, padx=5, expand=True)

    # mu
    frame = Frame(root)
    frame.pack(fill=X)
    mu0_label = Label(frame, text="Peak guess:", padx=5,
                      pady=5, width=20, justify=RIGHT, anchor="e")
    mu0_label.pack(side=LEFT)
    global mu0_var
    mu0_var = DoubleVar()
    mu0_entry = Entry(frame, textvariable=mu0_var)
    mu0_entry.pack(fill=X, padx=5, expand=True)

    # variance
    frame = Frame(root)
    frame.pack(fill=X)
    var0_label = Label(frame, text="Variance guess:",
                       padx=5, pady=5, width=20, justify=RIGHT, anchor="e")
    var0_label.pack(side=LEFT)
    global var0_var
    var0_var = DoubleVar()
    var0_entry = Entry(frame, textvariable=var0_var)
    var0_entry.pack(fill=X, padx=5, expand=True)

    # dx
    frame = Frame(root)
    frame.pack(fill=X)
    dx_label = Label(frame, text="X range:",
                     padx=5, pady=5, width=20, justify=RIGHT, anchor="e")
    dx_label.pack(side=LEFT)
    global dx_var
    dx_var = DoubleVar()

    dx_entry = Entry(frame, textvariable=dx_var)
    dx_entry.pack(fill=X, padx=5, expand=True)

    # f
    frame = Frame(root)
    frame.pack(fill=X)
    f_label = Label(frame, text="Weighing factor:",
                    padx=5, pady=5, width=20, justify=RIGHT, anchor="e")
    f_label.pack(side=LEFT)
    global f_var
    f_var = DoubleVar()

    f_entry = Entry(frame, textvariable=f_var)
    f_entry.pack(fill=X, padx=5, expand=True)

    #
    # Results
    #

    # Peak
    frame = Frame(root)
    frame.pack(fill=X)
    peak_label = Label(frame, text="Peak:",
                       padx=5, pady=5, width=20, justify=RIGHT, anchor="e")
    peak_label.pack(side=LEFT)
    global peak_var
    peak_var = StringVar()
    peak_var.set("")
    f_entry = Entry(frame, textvariable=peak_var, state="readonly")
    f_entry.pack(fill=X, padx=5, expand=True)

    set_defaults()
    root.mainloop()
