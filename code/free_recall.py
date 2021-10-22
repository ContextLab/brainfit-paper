# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import quail

import warnings


def get_bounds(data, bounds):
    data = [d for d in data.values.ravel() if not np.isnan(d)]
    return [np.percentile(data, p) for p in bounds]


def distplot(data, bounds, cmap, x='Performance', y='Number of participants', bins=15):
    bound_vals = get_bounds(data, bounds)
    data = [d for d in data.values.ravel() if not np.isnan(d)]

    sns.histplot(data, kde=True, color='k', bins=bins, edgecolor='w')
    plt.xlabel(x, fontsize=14)
    plt.ylabel(y, fontsize=14)
    ylim = plt.ylim()

    for i, p in enumerate(bound_vals):
        plt.plot([p, p], ylim, color=cmap[i], linewidth=4)

    plt.ylim(ylim);
    plt.xlim([0, 1]);

def create_fr_plots(fr, bounds, task_cmaps, prefix='', max_lag=8, save=False):
    # noinspection PyShadowingNames
    def get_fname(n, prefix):
        if len(prefix) == 0:
            return n + '.pdf'
        else:
            return f'{prefix}_{n}.pdf'

    # noinspection PyShadowingNames
    def group(vals, bounds):
        edges = np.percentile(vals, bounds)
        edges[-1] += 1
        return np.digitize(vals, edges).ravel() - 1

    # noinspection PyTypeChecker
    def plot_grouped_dists(data, groups, cmap, core_color='k', alpha1=0.25, alpha2=0.3, x='Feature',
                           y='Clustering score'):
        # noinspection PyShadowingNames
        def violin(x, color, alpha1=1, alpha2=0.5, linewidth=1):
            for i, c in enumerate(x.columns):
                parts = plt.violinplot([v for v in x[c].values if not np.isnan(v)], positions=[i])
                for p in parts['bodies']:
                    p.set_facecolor(color)
                    p.set_edgecolor(None)
                    p.set_alpha(alpha1 * alpha2)

                for j in ['cmeans', 'cmins', 'cmaxes', 'cbars', 'cmedians', 'cquantiles']:
                    if j not in parts.keys():
                        continue
                    parts[j].set_color(color)
                    parts[j].set_alpha(alpha1)
                    parts[j].set_linewidth(linewidth)

        unique_groups = np.unique(groups)
        for i, g in enumerate(unique_groups):
            violin(data.loc[groups == g], cmap[i], alpha1=alpha1, linewidth=1)
        violin(data, core_color, alpha1=alpha2, linewidth=2)
        plt.xticks(np.arange(len(data.columns)), [t.replace('_', '\n').capitalize() for t in data.columns.values])
        plt.xlabel(x, fontsize=14)
        plt.ylabel(y, fontsize=14)

    def plot_ribbon(data, color='k', alpha1=1, alpha2=0.25, linewidth=2):
        x = data.columns.values
        y = data.mean(axis=0)
        sem = np.divide(data.std(axis=0), np.sqrt(np.sum(1 - np.isnan(data.values), axis=0)))

        plt.fill_between(x, y + sem, y - sem, color=color, alpha=alpha1 * alpha2, edgecolor=None)
        plt.plot(x, y, c=color, linewidth=linewidth, alpha=alpha1)

    # noinspection PyUnusedLocal
    def plot_lines(data, color='k', opacity=0.05, linewidth=0.25):
        x = data.columns.values
        plt.plot(x, data.values.T, color=color, alpha=opacity, linewidth=linewidth)

    # noinspection PyTypeChecker
    def fr_summary_plot(data, groups, cmap, core_color='k', alpha1=0.5, alpha2=0.25, x='Serial position',
                        y='p(recall)'):
        unique_groups = np.unique(groups)
        for i, g in enumerate(unique_groups):
            # plotting individual curves looks messy; skip by default... (uncomment to show single-subject data)
            # plot_lines(data.loc[groups == g], cmap[i], opacity=alpha3)
            plot_ribbon(data.loc[groups == g], cmap[i + 1], alpha1=alpha1, alpha2=alpha2)
        plot_ribbon(data, core_color, alpha2=alpha2)
        plt.xlabel(x, fontsize=14)
        plt.ylabel(y, fontsize=14)

    # analyze data
    warnings.simplefilter('ignore')
    pfr = fr.analyze('pfr').data.groupby('Subject').mean()
    crp = fr.analyze('lagcrp').data.groupby('Subject').mean()
    spc = fr.analyze('spc').data.groupby('Subject').mean()
    fingerprint = fr.analyze('fingerprint').data.groupby('Subject').mean()
    accuracy = fr.analyze('accuracy').data.groupby('Subject').mean()
    fr_groups = group(accuracy.values, bounds)

    # plot accuracy (distribution)
    distplot(accuracy, bounds, task_cmaps['Free recall'], x='Accuracy')
    if save:
        plt.savefig(get_fname('accuracy', prefix))
    plt.clf()

    # plot fingerprints
    plot_grouped_dists(fingerprint, fr_groups, task_cmaps['Free recall'])
    if save:
        plt.savefig(get_fname('fingerprints', prefix))
    plt.clf()

    # probability of first recall
    fr_summary_plot(pfr, fr_groups, task_cmaps['Free recall'], y='Probability of first recall')
    if save:
        plt.savefig(get_fname('pfr', prefix))
    plt.clf()

    # lag-CRP
    lags = np.arange(-max_lag, max_lag + 1)
    fr_summary_plot(crp[lags], fr_groups, task_cmaps['Free recall'], x='Lag', y='Conditional response probability')
    if save:
        plt.savefig(get_fname('lag_crp', prefix))
    plt.clf()

    # serial position curve
    fr_summary_plot(spc, fr_groups, task_cmaps['Free recall'], y='Recall probability')
    if save:
        plt.savefig(get_fname('spc', prefix))
    plt.clf()


def split_by_feature(egg, feature):
    pres_items = egg.get_pres_items()
    pres_features = egg.get_pres_features()

    rec_items = egg.get_rec_items()
    rec_features = egg.get_rec_features()

    vals = np.unique([v[feature] for v in pres_features.values.ravel()])

    pres_words = []
    rec_words = []

    # subjs = pres_items.index.to_frame()['Subject'].values
    current_id = pres_items.index[0][0]
    subj_pres_words = []
    subj_rec_words = []
    for i in range(pres_items.shape[0]):
        next_id = pres_items.index[i][0]
        if next_id != current_id:
            pres_words.append(subj_pres_words)
            rec_words.append(subj_rec_words)

            subj_pres_words = []
            subj_rec_words = []

            current_id = next_id

        for v in vals:
            # get next presented and recalled words
            next_pres_items = [dw.core.update_dict(x, {'item': w}) for w, x in
                               zip(pres_items.iloc[i], pres_features.iloc[i]) if x[feature] == v]
            next_rec_items = [dw.core.update_dict(x, {'item': w}) for w, x in
                              zip(rec_items.iloc[i], rec_features.iloc[i]) if w in [y['item'] for y in
                                                                                    next_pres_items]]

            # remove "feature" from pres and rec items
            [x.pop(feature, None) for x in next_pres_items]
            [x.pop(feature, None) for x in next_rec_items]

            subj_pres_words.append(next_pres_items)
            subj_rec_words.append(next_rec_items)

    try:
        pres_words.append(subj_pres_words)
        rec_words.append(subj_rec_words)
    except NameError:
        pass

    return quail.Egg(pres=pres_words, rec=rec_words)