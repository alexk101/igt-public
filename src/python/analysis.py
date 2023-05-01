import polars as pl
from pathlib import Path
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from scipy.stats import entropy
from collections import defaultdict
from utils import *
from typing import List, Dict
import networkx as nx
from PIL import Image
from simulation import generate_sim_plots

dir_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_path)


def plot_data(name: str, period: str, samples: int):
    if (OUTPUT / 'indiv' / f'{name}_{period}.png').exists():
        print(f"{OUTPUT / 'indiv' / f'{name}_{period}.png'} exists already. \nSkipping.")
        return

    sig = pl.scan_csv(SIGNALS_PATH / f'{name}.csv')
    sig = sig.filter(pl.col('period') == period)
    print(f'Plotting {name} for {period}')
    data_len = samples
    avgs = sig.groupby('participant').agg([pl.col(f'data_{x}').mean().alias(f'{x}') for x in range(data_len)])
    
    std = sig.groupby('participant').agg([pl.col(f'data_{x}').std().alias(f'{x}') for x in range(data_len)])
    std = std.sort('participant')

    avgs = avgs.collect()
    std = std.collect()
    if not avgs.height:
        print(f'no samples for {name} filtering {period}')
        return

    # Get subjects
    sub = std.to_numpy()[:,:1].astype(int)
    avgs = avgs.sort('participant').to_numpy()[:,1:].astype(float)
    std = std.to_numpy()[:,1:].astype(float)

    ci_u = []
    ci_l = []

    for x in range(len(std)):
        sem = (0.95 * (std[x]/np.sqrt(N)))
        ci_u.append(avgs[x] + sem)
        ci_l.append(avgs[x] - sem)

    ci_u = np.array(ci_u)
    ci_l = np.array(ci_l)

    time = np.linspace(0,len(avgs[0]), len(avgs[0]), endpoint=False)/60
    fig, ax = plt.subplots(
        int(np.ceil(len(avgs)/10)), R_N, 
        figsize=(40,40), sharex=True, sharey=True
    )
    ax = np.array(ax)
    col : plt.Axes

    if len(ax.shape) != 1:
        for r_ind, row in enumerate(ax):
            for col_ind, col in enumerate(row):
                ind = (r_ind * R_N) + col_ind
                if ind >= len(avgs):
                    break

                col.plot(time, avgs[ind])
                col.fill_between(time, ci_l[ind], ci_u[ind], color='b', alpha=.1)
                col.set_title(f'Subject {sub[ind]}')
    else:
        for col_ind, col in enumerate(ax):
            ind = col_ind
            if ind >= len(avgs):
                break

            col.plot(time, avgs[ind])
            col.fill_between(time, ci_l[ind], ci_u[ind], color='b', alpha=.1)
            col.set_title(f'Subject {sub[ind]}')

    fig.savefig(str(OUTPUT / 'indiv' / f'{name}_{period}.png'))
    plt.close(fig)


def plot_groups(name: str, period: str, samples: int, freq: int, group: str, ax: plt.Axes):
    sig = import_signal(name, period)


    # Calculate overall averages and standard deviations
    total_avgs = sig.select([pl.col(f'data_{x}').mean().alias(f'{x}') for x in range(samples)]).to_numpy()[0].astype(float)
    total_std = sig.select([pl.col(f'data_{x}').std().alias(f'{x}') for x in range(samples)]).to_numpy()[0].astype(float)

    # Calculate lower and upper confidence intervals
    ci_u_total = []
    ci_l_total = []

    for x in range(len(total_std)):
        sem = (0.95 * (total_std[x]/np.sqrt(len(set(total_avgs['participants'])))))
        ci_u_total.append(total_avgs[x] + sem)
        ci_l_total.append(total_avgs[x] - sem)

    ci_u_total = np.array(ci_u_total)
    ci_l_total = np.array(ci_l_total)

    # Calculate group averages and standard deviations
    avgs = sig.groupby(group).agg([pl.col(f'data_{x}').mean().alias(f'{x}') for x in range(samples)])
    std = sig.groupby(group).agg([pl.col(f'data_{x}').std().alias(f'{x}') for x in range(samples)])

    avgs = avgs.sort(group).to_numpy()
    avgs = dict(zip(avgs[:,:1].flatten(), avgs[:,1:].astype(float)))
    std = std.sort(group).to_numpy()
    std = dict(zip(std[:,:1].flatten(), std[:,1:].astype(float)))
    
    if None in avgs.keys():
        avgs.pop(None)
        std.pop(None)

    # Plot
    time = np.linspace(0, samples, samples, endpoint=False) / freq

    # Overall
    ax.plot(time, total_avgs, label='Overall')
    ax.fill_between(time, ci_l_total, ci_u_total, alpha=.1) # type: ignore

    for ind, group_name in enumerate(avgs.keys()):

         # Calculate lower and upper confidence intervals
        ci_u = []
        ci_l = []

        for x in range(len(std[group_name])):
            sem = (0.95 * (std[group_name][x]/np.sqrt(N)))
            ci_u.append(avgs[group_name][x] + sem)
            ci_l.append(avgs[group_name][x] - sem)

        ci_u = np.array(ci_u)
        ci_l = np.array(ci_l)

        ax.plot(time, avgs[group_name], label=group_name.capitalize())
        ax.fill_between(time, ci_l, ci_u, alpha=.1) # type: ignore
        ax.set_title(f'{group_name}')
    ax.set_title(period)
    ax.legend()
    sizes = sig.groupby(group).agg(pl.count()//100)
    sizes = dict(zip(sizes[group], sizes['count']))
    # Sort groups by name and remove None if present
    if None in sizes:
        sizes.pop(None)
    sizes = {key: val for key, val in sorted(sizes.items(), key=lambda x: x[0])}
    return sizes


def plot_group_quintiles(name: str, period: str, samples: int, freq: int, group: str, axes: List[plt.Axes]):
    
    sig = import_signal(name, period)
    group_set = sig[group].unique().to_list()
    if None in group_set:
        group_set.remove(None)
    group_set = sorted(group_set)
    assert len(axes) == len(group_set)

    time = np.linspace(0, samples, samples, endpoint=False) / freq
    data = defaultdict(dict)
    for ind, (ax, group_label) in enumerate(zip(axes, group_set)):
        group_data = sig.filter(pl.col(group) == group_label)
        for quintile in range(0,5):
            quintile_data = group_data.filter((pl.col('trial') >= (quintile*20)) & (pl.col('trial') < ((quintile+1)*20)))

            # Calculate overall averages and standard deviations
            quintile_avgs = quintile_data.select([pl.col(f'data_{x}').mean().alias(f'{x}') for x in range(samples)]).to_numpy()[0].astype(float)
            quintile_std = quintile_data.select([pl.col(f'data_{x}').std().alias(f'{x}') for x in range(samples)]).to_numpy()[0].astype(float)

            # Calculate lower and upper confidence intervals
            ci_u = []
            ci_l = []

            for x in range(len(quintile_std)):
                sem = (0.95 * (quintile_std[x]/np.sqrt(N)))
                ci_u.append(quintile_avgs[x] + sem)
                ci_l.append(quintile_avgs[x] - sem)

            ci_u = np.array(ci_u)
            ci_l = np.array(ci_l)
            data[group][quintile] = {
                'ci_u': ci_u,
                'ci_l': ci_l,
                'avg': quintile_avgs
            }

            if ind == (len(group_set)-1):
                ax.plot(time, quintile_avgs, label=f'Q{quintile+1}')
            else:
                ax.plot(time, quintile_avgs)

            ax.fill_between(time, ci_l, ci_u, alpha=.1) # type: ignore
        ax.set_title(f'{group_label.capitalize()}')


def deck_choice_plot(group: str):
    out_folder = Path('./plots/decks')
    out_folder.mkdir(parents=True, exist_ok=True)
    output = out_folder / f'{group.replace("/","_")}_deck.png'
    if output.exists():
        print(f'{output} exists. \nSkipping.')

    groups = pl.read_csv(INDIV_PATH / 'groups.csv')
    groups = groups.rename({'exp': 'participant'}).with_columns(pl.col('participant').cast(pl.Int32))
    possiblities = groups[group].unique().to_list()
    possiblities.append('overall')
    if None in possiblities:
        possiblities.remove(None)

    trials = pl.read_csv(INDIV_PATH / f'igt_trial_info.csv')

    new_trials = []
    for subject in groups['participant'].unique():
        sub_dat = trials.filter(pl.col('participant') == subject)
        sub_grps = groups.filter(pl.col('participant') == subject).to_dicts()[0]
        sub_grps.pop('participant')
        if sub_grps['ve_median_split'] is None:
            sub_grps['ve_median_split'] = ''
        sub_dat = sub_dat.with_columns([pl.lit(val).alias(key) for key, val in sub_grps.items()])
        new_trials.append(sub_dat)
    trials = pl.concat(new_trials)

    quintiles = []
    for quintile in range(5):
        data = trials.filter((pl.col('trial') >= (quintile*20)) & (pl.col('trial') < ((quintile + 1)*20)))
        data = data.filter((pl.col(group) != '') & (pl.col(group) != None))
        quintile_data = data.groupby(group,'deck').agg(pl.count())
        overall = data.groupby('deck').agg(pl.count()).sort('deck')
        overall = overall.with_columns([pl.lit('overall').alias(group)])
        quintile_data = pl.concat([quintile_data,overall], how='diagonal')

        counts = quintile_data.groupby(group).agg(pl.sum('count'))

        norm = []
        # Normalize
        for x in possiblities:
            grp = quintile_data.filter(pl.col(group) == x)
            count = counts.filter(pl.col(group) == x).to_numpy()[0][-1]
            grp = grp.with_columns(pl.col('count') / count)
            norm.append(grp)
        norm = pl.concat(norm)
        quintile_data = norm

        deck_tree = defaultdict(dict)
        for grp in possiblities:
            for deck in trials['deck'].unique():
                deck_tree[grp][deck] = quintile_data.filter((pl.col('deck') == deck) & (pl.col(group) == grp)).to_numpy()[0][-1]

        advan = defaultdict(list)
        for grp in possiblities:
            advan['group'].append(grp)
            advan['group'].append(grp)
            advan['choice'].append('advantage')
            advan['choice'].append('disadvantage')
            advan['count'].append(deck_tree[grp]['DeckC'] + deck_tree[grp]['DeckD'])
            advan['count'].append(deck_tree[grp]['DeckA'] + deck_tree[grp]['DeckB'])
        advan = pl.DataFrame(advan).with_columns([pl.lit(quintile).alias('quintile')])
        quintiles.append(advan)
    quintiles = pl.concat(quintiles).to_pandas()
    g = sns.relplot(data=quintiles, x="quintile", y="count", hue="choice", col="group", col_wrap=3, kind='line')
    g.set(ylim=(0, 1))
    g.figure.savefig(str(output))
    plt.close(g.figure)


def deck_entropy(group: str):
    output_folder = Path('./plots/entropy')
    output_folder.mkdir(exist_ok=True, parents=True)
    grp_name = group.replace("/","_")
    if (output_folder / f'{grp_name}_deck_entropy.png').exists() and (output_folder / f'{grp_name}_adv_entropy.png').exists():
        print(f'Entropy for group {group} has already been plotted. \nSkipping.')
        return

    data = pl.read_csv(INDIV_PATH / 'igt_trial_info.csv')
    groups = pl.read_csv(INDIV_PATH / 'groups.csv')
    possiblities = groups[group].unique().to_list()
    if None in possiblities:
        possiblities.remove(None)
    members = {x: list(groups.filter(pl.col(group) == x)['exp'].to_numpy()) for x in possiblities}

    decks = set(data['deck'])

    deck_entropy = defaultdict(list)
    for subject in data['participant'].unique():
        for quintile in range(5):

            sub_data = data.filter(
                (pl.col('participant') == subject) & 
                (pl.col('trial') >= quintile*20) & 
                (pl.col('trial') < (quintile+1)*20)
            )
            counts = sub_data.groupby('deck').agg(pl.count())
            diff = decks.difference(counts['deck'].unique())
            if len(diff):
                frame = pl.DataFrame(
                    {
                        'deck': list(diff), 
                        'count': [0]*len(diff)
                    }, 
                    schema={'deck': pl.Utf8, 'count': pl.UInt32}
                )
                counts = pl.concat([counts, frame])

            counts = counts.with_columns([pl.col('count') / counts['count'].sum()])
            probs = counts['count'].to_numpy()

            deck_entropy['subject'].append(subject)
            deck_entropy['quintile'].append(quintile+1)
            deck_entropy['entropy'].append(entropy(probs))
            deck_entropy['type'].append('deck')

            deck_entropy['subject'].append(subject)
            deck_entropy['quintile'].append(quintile+1)
            decks_probs = dict(zip(counts['deck'], counts['count']))
            adv_probs = [decks_probs['DeckC']+decks_probs['DeckD']]
            adv_probs += [decks_probs['DeckA']+decks_probs['DeckB']]
            deck_entropy['entropy'].append(entropy(adv_probs))
            deck_entropy['type'].append('adv')

    all_entropy = pl.DataFrame(deck_entropy)
    deck_entropy = all_entropy.filter(pl.col('type') == 'deck')
    adv_entropy = all_entropy.filter(pl.col('type') == 'adv')
    entropies = {
        'deck': deck_entropy, 
        'adv': adv_entropy
    }

    for key, entrop in entropies.items():
        g = sns.FacetGrid(entrop.to_pandas(), col="subject", col_wrap=10)
        g.map_dataframe(sns.barplot, x="quintile", y='entropy')
        g.figure.suptitle(f'{key.capitalize()} Choice Entropy by Quintile')
        filename = str(output_folder / f'{key}_entropy.png')
        g.figure.savefig(filename)
        plt.close(g.figure)

        fig, ax = plt.subplots(1,1+len(members), figsize=(20,10), sharex=True, sharey=True)

        # Overall Entropy
        fig.suptitle(f'{key.capitalize()} Entropy')
        ax[0].set_title('Overall')
        sns.boxplot(data=entrop.to_pandas(), x="quintile", y="entropy", ax=ax[0])

        for ind, (group_name, subs) in enumerate(members.items()):
            # Winner Entropy
            ax[ind+1].set_title(group_name.capitalize())
            group_data = entrop.filter(pl.col('subject').is_in(subs)) # type: ignore
            sns.boxplot(data=group_data.to_pandas(), x="quintile", y="entropy", ax=ax[ind+1])

        fig.tight_layout()
        filename = str(output_folder / f'{group.replace("/","_")}_{key}_entropy.png')
        fig.savefig(filename)
        plt.close(fig)


def indiv_diff(group: str):
    # ... do something ...
    grp_name = group.replace("/","_")
    output_folder = Path('./plots/factors')
    output_folder.mkdir(parents=True, exist_ok=True)
    output = output_folder / f'{grp_name}_factors.png'
    if output.exists():
        print(f'Individual factors for group {group} plotted. \nSkipping.')
        return

    subs, factors = import_factors(group)

    g = sns.PairGrid(factors.to_pandas(), hue=group)
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    g.add_legend()
    g.figure.savefig(str(output))
    plt.close(g.figure)


def plot_overall(group: str):
    for signal, freq in SIGNALS.items():
        sizes = {}
        fig, axes = plt.subplots(len(PERIODS), 1, figsize=(10,4*len(PERIODS)))
        fig.suptitle(f'Average Overall {signal}: {group}')
        group_out = OUTPUT / 'group' / NAME_MAP[group.replace("/","_")] / 'overall'
        group_out.mkdir(exist_ok=True, parents=True)
        outfile = group_out / f'{signal}.png'
        if outfile.exists():
            print(f"{outfile} exists already. \nSkipping.")
            plt.close(fig)
            continue
        print(f'Plotting {outfile}')
        for (period, time), ax in zip(PERIODS.items(), axes):
            plot_groups(signal, period, time*freq, freq, group, ax)

        n = '\n'.join([f'{key}: n={val}' for key,val in sizes.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        fig.text(x=0.05, y=0.95, s=n, bbox=props)
        fig.tight_layout()
        fig.savefig(str(outfile))
        plt.close(fig)


def more_plot(name: str, freq: int, group: str):

    fuck = []
    for period in ['baseline', 'planning', 'feedback']:
        seconds = PERIODS[period]

        sig = import_signal(name, period)
        group_set = sig[group].unique().to_list()
        if None in group_set:
            group_set.remove(None)
        group_set = sorted(group_set)

        quintile_data = []
        for quintile in range(0,5):
            quint = sig.filter((pl.col('trial') >= (quintile*20)) & (pl.col('trial') < ((quintile+1)*20)))
            quint = quint.with_columns([pl.lit(f'Q{quintile+1}').alias('quintile')])
            quintile_data.append(quint)
        sig = pl.concat(quintile_data)
        id_vars = ['participant', 'quintile', group, 'period']
        data_vars = [f'data_{x}' for x in range(seconds*freq)]
        sig = sig.select(id_vars+data_vars)
        sig = sig.melt(id_vars, variable_name='time', value_name='value')
        sig = sig.with_columns([pl.col('time').str.slice(5).cast(pl.Float64) / freq])
        fuck.append(sig)

    sig = pl.concat(fuck)
    g = sns.relplot(sig.to_pandas(), x='time', y='value', col='quintile', hue=group, kind='line', row='period')
    return g


def plot_quintiles(group: str):
    for signal, freq in SIGNALS.items():
        group_out = OUTPUT / 'group' / NAME_MAP[group.replace("/","_")] / 'quintiles'
        group_out.mkdir(exist_ok=True, parents=True)
        outfile = group_out / f'{signal}.png'
        if outfile.exists():
            print(f"{str(outfile)} exists already. \nSkipping.")
            continue
        print(f'Plotting {str(outfile)}')
        g = more_plot(signal, freq, 'win/loss_2')

        g.fig.savefig(str(outfile))
        plt.close(g.fig)


def plot_process(counts: Dict[str, Dict[str,float]], group: str, ax: plt.Axes, output_folder: Path) -> None:
    grp_name = group.replace("/","_")
    G = nx.DiGraph()
    for start in counts:
        for end in counts[start]:
            G.add_edge(start, end, label=f"{int((counts[start][end])*100)}%")

    A = nx.nx_agraph.to_agraph(G)  # convert to a graphviz graph
    A.layout(prog='dot')
    temp_file = str(output_folder / f'{grp_name}.png')
    A.draw(temp_file)
    graph = np.asarray(Image.open(temp_file))
    os.remove(temp_file)
    ax.imshow(graph)
    ax.set_axis_off()


def markov(group: str):
    output_folder = Path('./plots/markov/Real Data')
    grp_name = group.replace("/","_")

    if (output_folder / NAME_MAP[grp_name]).exists():
        print(f"Skipping {str(output_folder / NAME_MAP[grp_name])}")
        return
    trials = import_trial_data()
    

    group_set = trials[group].unique().to_list()
    if '' in group_set:
        group_set.remove('')
    group_set = list(group_set)

    adv_map = {
        'DeckA': 'Disadvantage',
        'DeckB': 'Disadvantage',
        'DeckC': 'Advantage',
        'DeckD': 'Advantage'
    }

    types = {'All Decks': trials, 'Advantageous vs Disadvantageous': trials.with_columns(pl.col('deck').map_dict(adv_map))}

    for bruh, bruh_data in types.items():
        all_decks = bruh_data['deck'].unique().to_list()
        # Overall Plots
        fig, axes = plt.subplots(1, len(group_set), figsize=(5*len(group_set),5))
        (output_folder / NAME_MAP[grp_name] / bruh / 'process').mkdir(exist_ok=True, parents=True)
        (output_folder / NAME_MAP[grp_name] / bruh / 'bar plots').mkdir(exist_ok=True, parents=True)

        overall_markov = defaultdict(list)
        for member, ax in zip(group_set, axes):
            group_data = bruh_data.filter(pl.col(group) == member)
            probs = TransitionMatrix(group_data, all_decks).to_dict()
            plot_process(probs, group, ax, output_folder)
            ax.set_title(f'{member}')
            for key, val in probs.items():
                for deck, prob in val.items():
                    overall_markov['group'].append(member)
                    overall_markov['from'].append(key)
                    overall_markov['to'].append(deck)
                    overall_markov['chance'].append(prob)
        overall_markov = pl.DataFrame(overall_markov)

        fig.suptitle(f'Markov Process Diagram for {group}- Overall')
        fig.savefig(str(output_folder / NAME_MAP[grp_name] / bruh / 'process' / f'overall.png'))
        
        # Quintile Plots
        fig = plt.figure(layout='constrained', figsize=(9*len(group_set),10))
        fig.suptitle(f'Markov Process Diagram for {group}- Quintiles')
        subfigs = fig.subfigures(len(group_set), 1, wspace=0.07)


        quintile_markov = defaultdict(list)
        for member, subfig in zip(group_set, subfigs):
            print(f'member: {member}')
            group_data = bruh_data.filter(pl.col(group) == member)
            subfig.suptitle(member)
            axes = subfig.subplots(1,5)
            for quintile, ax in zip(range(0,5), axes):
                quintile_data = group_data.filter((pl.col('trial') >= (quintile*20)) & (pl.col('trial') < ((quintile+1)*20)))
                probs = TransitionMatrix(quintile_data, all_decks).to_dict()
                plot_process(probs, group, ax, output_folder)
                ax.set_title(f'Q{quintile+1}')
                for key, val in probs.items():
                    for deck, prob in val.items():
                        quintile_markov['group'].append(member)
                        quintile_markov['quintile'].append(quintile+1)
                        quintile_markov['from'].append(key)
                        quintile_markov['to'].append(deck)
                        quintile_markov['chance'].append(prob)
        quintile_markov = pl.DataFrame(quintile_markov)
        
        fig.suptitle(f'Markov Process Diagram for {group}- Quintiles')
        fig.savefig(str(output_folder / NAME_MAP[grp_name] / bruh / 'process' / f'quintiles.png'))
        plt.close(fig)

        g = sns.catplot(
            data=overall_markov.to_pandas(), 
            x="from", 
            y="chance", 
            hue='to', 
            row="group",
            kind="bar"
        )
        g.figure.savefig(str(output_folder / NAME_MAP[grp_name] / bruh / 'bar plots' / f'overall.png'))

        g = sns.catplot(
            data=quintile_markov.to_pandas(), 
            x="from", 
            y="chance", 
            hue='to', 
            col="quintile",
            row='group',
            kind="bar"
        )
        g.figure.savefig(str(output_folder / NAME_MAP[grp_name] / bruh / 'bar plots' / f'quintiles.png'))
        plt.close(g.figure)


def ttp(group: str):
    paths = ['Winning Trials vs Losing Trials', 'overall', 'quintiles']
    grp_name = group.replace("/","_")
    output_folder = Path(f'./plots/latency/{NAME_MAP[grp_name]}')

    for path in paths:
        (output_folder / path).mkdir(exist_ok=True, parents=True)

    periods = ['baseline', 'planning', 'feedback']

    for signal in SIGNALS.keys():
        if (output_folder / f'{signal}.png').exists():
            continue        

        fig = plt.figure(layout='constrained', figsize=(10,5*len(periods)))
        fig.suptitle(f'{signal} Latency for {grp_name} - Trial Type')
        subfigs = fig.subfigures(3, 1, wspace=0.07)

        fig_2 = plt.figure(layout='constrained', figsize=(10,5*len(periods)))
        fig_2.suptitle(f'{signal} Latency for {grp_name} - Overall')
        subfigs_2 = fig_2.subfigures(3, 1, wspace=0.07)

        fig_3 = plt.figure(layout='constrained', figsize=(10,5*len(periods)))
        fig_3.suptitle(f'{signal} Latency for {grp_name} - Quintiles')
        subfigs_3 = fig_3.subfigures(3, 1, wspace=0.07)

        all_figs = (fig, fig_2, fig_3)
        all_subfigs = [(x,y,z) for x, y, z in zip(subfigs, subfigs_2, subfigs_3)]

        for ind, (subfig_group, period) in enumerate(zip(all_subfigs, periods)):
            subfig, subfig_2, subfig_3 = subfig_group

            for oof in subfig_group:
                oof.suptitle(period)

            axes = subfig.subplots(1, 2, sharey=True)
            axes_2 = subfig_2.subplots(1, 2, sharey=True)
            axes_3 = subfig_3.subplots(1, 2, sharey=True)
            all_axes = [axes, axes_2, axes_3]

            if not ind:
                for oof in all_axes:
                    oof[0].set_title('Max Latency')
                    oof[1].set_title('Min Latency')

            ttp_data = import_ttp(signal, period).rename({'net_gain':'trial type'})
            ttp_data = ttp_data.filter(pl.col(group) != '')
            ttp_data = ttp_data.with_columns(((pl.col('trial') // 20)+1).alias('quintile')).to_pandas()

            # Trial Type
            # Min latency
            sns.boxplot(ttp_data, x=group, y='max', hue='trial type', ax=axes[0])
            axes[0].legend().remove()

            # Max latency
            sns.boxplot(ttp_data, x=group, y='min', hue='trial type', ax=axes[1])

            # Overall
            # Max latency
            sns.boxplot(ttp_data, x=group, y='max', ax=axes_2[0])

            # Min latency
            sns.boxplot(ttp_data, x=group, y='min', ax=axes_2[1])

            # Quintiles
            # Max latency
            sns.boxplot(ttp_data, x=group, y='max', hue='quintile', ax=axes_3[0])
            axes_3[0].legend().remove()

            # Max latency
            sns.boxplot(ttp_data, x=group, y='min', hue='quintile', ax=axes_3[1])

        fig.savefig(str(output_folder / 'Winning Trials vs Losing Trials' / f'{signal}.png'))
        fig_2.savefig(str(output_folder / 'overall' / f'{signal}.png'))
        fig_3.savefig(str(output_folder / 'quintiles' / f'{signal}.png'))
        for oof in all_figs:
            plt.close(oof)


def plot_score(scores: pl.DataFrame, group: str, output: Path=Path('./plots/score'), title: str=''):
    output.mkdir(exist_ok=True, parents=True)
    grp_name = group.replace("/","_")
    if (output / f'{grp_name}.png').exists():
        print(f'Skipping score plot for {grp_name}')
        return

    if group == 've_median_split':
        scores = scores.filter(pl.col(group) != '')

    g = sns.lineplot(scores.to_pandas(), x='trial', y='score', hue=group)
    if title == '':
        g.figure.suptitle(f'{grp_name} Scores')
    else:
        g.figure.suptitle(title)
    g.figure.savefig(str(output / f'{grp_name}.png'))
    plt.close(g.figure)


def main() -> int:
    groups = ['win/loss_2', 've_median_split', 'net_gain']

    for signal, freq in SIGNALS.items():
        for period, time in PERIODS.items():
            plot_data(signal, period, time*freq)

    scores = import_trial_data()
    plot_score(scores, 'participant')
    for group in groups:
        plot_overall(group)
        plot_quintiles(group)
        if group != 'net_gain':
            plot_score(scores, group)
            ttp(group)
            deck_entropy(group)
            deck_choice_plot(group)
            indiv_diff(group)
            markov(group)
    generate_sim_plots()

    return 0


if __name__ == '__main__':
    sys.exit(main())