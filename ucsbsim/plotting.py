
def quick_plot(
        ax,
        x,
        y,
        labels=None,
        title=None,
        xlabel=None,
        ylabel=None,
        ylim=None,
        twin=None,
        first=False,
        **kwargs):
    """
    Condenses matplotlib plotting into one line (generally).
    :param ax: axes for plotting
    :param x: x axis data
    :param y: y axis data, dimensions must match x
    :param labels: labels for each set of data
    :param title: title of subplot
    :param xlabel: x axis label
    :param ylabel: y axis label
    :param ylim: y axis upper limit (convenient for twin)
    :param twin: y axis twin color, if any
    :param first: True if first line for plot, will make grid
    :param kwargs: color, linestyle, markerstyle, etc.
    :return: the desired plot (not shown, use plt.show() after)
    """
    if first:
        ax.grid()
    if labels is None:
        labels = ['_nolegend_' in range(len(x))]
    for n, (w, f) in enumerate(zip(x, y)):
        ax.plot(w, f, label=labels[n], **kwargs)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(top=ylim)
    ax.legend()
    if twin is not None:
        ax.tick_params(axis="y", labelcolor=twin)
        ax.yaxis.label.set_color(twin)
        ax.spines['right'].set_color(twin)
        ax.legend(loc='lower right', labelcolor=twin)


def simulation_plot(
        n_axes: int,
        all_title: str,
        xlims,
        data,
        labels,
        titles,
        ylabels,
        xlabel: str,
        colors,
        twin,
        save_loc: str,
):
    """
    :param int n_axes: number of plots totals
    :param str all_title: title for entire figure
    :param xlims: left/right limits for x-axis as list
    :param data: n_axes lists with each containing plots for each axes
    :param labels: n_axes lists with each containing labels for each plot
    :param titles: 1D list of n_axes titles
    :param ylabels: 1D list of n_axes y-axis labels
    :param str xlabel: x-axis label for entire figure (sharing axis)
    :param colors: n_axes lists with each containing colors for each plot
    :param twin: list containing [axis, color, data, label, ylabel]
    :param str save_loc: directory/file_name for saving
    :return: plot shown and saved as 8.5" x 11" pdf
    """
    fig, axes = plt.subplots(n_axes, 1, sharex=True, figsize=(8.5, 11), dpi=300)
    axes = axes.ravel()
    plt.suptitle(all_title, fontweight='bold')

    for p in range(n_axes):
        axes[0].set_xlim(lims)
        axes[p].set_title(titles[p])
        axes[p].set_ylabel(ylabels[p])
        quick_plot(axes[p], data[p][0], data[p][1], labels=labels[p], color=colors[p])
        axes[-1].set_xlabel(xlabel)

    quick_plot(axes[1], lambda_pixel, convol_summed, color='red',
               labels=["Conv+Int"] + ['_nolegend_' for o in range(nord - 1)], first=True)
    quick_plot(axes[1], lambda_pixel, direct_flux_calc, color='k',
               labels=[r"$\lambda$2Pix Int."] + ['_nolegend_' for o in range(nord - 1)],
               title=r"Convolved+Integrated & $\lambda$-to-Pixel Integrated",
               ylabel=r"Flux (phot $cm^{-2} s^{-1})$")

    # plotting comparison between final counts and convolved-integrated spectrum:
    quick_plot(axes[2], lambda_pixel, photons_binned[::-1], ylabel="Photon Count", color='k',
               labels=['Observed+FSR-Binned'] + ['_nolegend_' for o in eng.spectrograph.orders[:-1]], first=True)
    twin = axes[2].twinx()
    quick_plot(twin, lambda_pixel, convol_summed, ylabel=r"Flux (phot $cm^{-2} s^{-1})$", color='red',
               labels=['Conv.+Int.'] + ['_nolegend_' for o in eng.spectrograph.orders[:-1]],
               title="Convolved+Integrated vs. Observed Photons", xlabel="Wavelength (nm)", twin='red', alpha=0.5)
    fig.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.6, left=0.1)
    plot_file = f'output_files/{type_of_spectra}/intplots_R0{eng.spectrograph.detector.design_R0}.pdf'
    fig.savefig(plot_file)
    logging.info(f'\nSaved intermediate plots to {plot_file}.')
    plt.show()