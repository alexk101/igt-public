import sys
import polars as pl
import seaborn as sns
from utils import *
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm # type: ignore
from matplotlib.figure import SubFigure
from typing import Iterable
from PIL import Image
from JaxPCA import PCA
from sklearn.manifold import TSNE
import pandas as pd
from umap import UMAP
from simulation import *

dir_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_path)

def plot_radar(components: pl.DataFrame, period: str, subfig: SubFigure, ex_var: np.ndarray, labels: Iterable, wrap: bool=False, time: bool=False):    
    # Initialise the spider plot by setting figure size and polar projection
    axes = subfig.subplots(1, components.height, subplot_kw=dict(projection='polar'), sharey=True)
    subfig.suptitle(f'Period: {period.capitalize()} -> Total Explained Variance: {ex_var.sum()*100:.2f}%')
    
    y_num = len(components.columns)
    if wrap:
        y_num += 1
    theta = np.linspace(0, 2 * np.pi, y_num)
    colors = cm.rainbow(np.linspace(0, 1, components.height))

    # Arrange the grid into number of sales equal parts in degrees

    for ind, (component, c, ax, var) in enumerate(zip(components.iter_rows(), colors, axes, ex_var)):
        ax.set_thetagrids(range(0, 360, int(360/len(labels))), labels) # type: ignore
        ax.set_title(f'PC {ind+1} -> {var*100:.2f}%')
        if wrap:
            component = list(component)+ [component[0]]
        ax.plot(theta, component, c=c)
        ax.fill(theta, component, c=c, alpha=0.1)


def dimred_sig(zscore: bool = False, rpca: bool = False):
    qual = ''
    if zscore:
        qual = '_zscored'
    pca_out = Path('pca') / 'individual'
    sigs = ['pupil', 'ad_pupil', 'rsa', 'hr', 'rsp']
    rpca_sigs = ['pupil', 'ad_pupil']

    # PCA
    n_comp = 5

    for sig in sigs:
        if (pca_out / sig).exists():
            pass
        output = []
        tsne_out = []
        fig, row = plt.subplots(len(PERIODS), 1, sharex=True, sharey=True)
        fig.suptitle(sig)
        (pca_out / sig).mkdir(exist_ok=True, parents=True)
        
        # radar plots
        radar = plt.figure(layout='constrained', figsize=(9*n_comp, 6*len(PERIODS)))
        radar.suptitle(f'PCA Loadings {sig}: Top {n_comp} PC')
        sub_radars = radar.subfigures(len(PERIODS), 1, wspace=0.07)

        for ax, sub_radar, period in zip(row, sub_radars, PERIODS):
            ax.set_title(period)
            clean = (sig in rpca_sigs) and rpca
            data = import_signal(sig, period, zscore, rpca)
            subs = data['participant']
            trials = data['trial']
            data = data.select([f'data_{x}' for x in range(PERIODS[period]*SIGNALS[sig])])
            # impute mean of column
            # data = fill_null(data, period, sig)

            # Backwards / Forwards fills
            data = data.fill_null(strategy='backward')
            data = data.fill_null(strategy='forward')

            # TSNE
            dim_model = TSNE(
                n_components=2,
                n_iter=1000,
                n_iter_without_progress=150,
                n_jobs=2,
                random_state=0,
            )
            dim_red = dim_model.fit_transform(data.to_numpy())
            dim_red = pl.from_numpy(dim_red)
            dim_red = dim_red.with_columns([subs, trials, pl.lit(period).alias('period')])
            tsne_out.append(dim_red)

            # PCA
            # Normalize
            data = data.with_columns([(pl.col(x) - pl.col(x).min())/(pl.col(x).max()-pl.col(x).min()) for x in data.columns])

            pca = PCA(data.to_numpy(), n_comp) # type: ignore
            red = pca.fit_transform()
            red = pl.from_numpy(np.array(red))
            
            red = red.with_columns([subs, trials, pl.lit(period).alias('period')])
            output.append(red)

            components = pl.from_numpy(np.array(pca.loadings))
            ex_var = np.array(pca.ex_var) / 100

            n_labels = 2 * PERIODS[period]
            labels = np.linspace(0, PERIODS[period], n_labels+1, endpoint=True).astype(str)[:-1]
            plot_radar(components, period, sub_radar, ex_var[:5], labels)
            
            print(f'First 5 ({sig}, {period}): {ex_var[:5].sum()*100:.2f}%')
            comps = list(range(len(ex_var)))
            ax.plot(comps, ex_var)
        tsne_out = pl.concat(tsne_out).sort('participant', 'trial', 'period')
        output = pl.concat(output).sort('participant', 'trial', 'period')
        radar.savefig(str(pca_out / sig / f'loadings_{qual}.png'))
        plt.close(radar)
        output.write_csv(str(pca_out / sig  / f'{sig}_pca.csv'))
        tsne_out.write_csv(str(pca_out / sig  / f'{sig}_tsne.csv'))
        fig.savefig(str(pca_out / sig / f'explained_variance_{qual}.png'))
        plt.close(fig)
    return


def dim_red_multisig(zscore: bool = False, rpca: bool = False):
    qual = '_standard'
    if zscore:
        qual = '_zscored'
    if rpca:
        qual = '_rpca'
    if zscore and rpca:
        qual = '_both'
    pca_out = Path('pca')
    combs = []
    combs.append(['pupil', 'hr', 'rsa', 'rsp'])
    combs.append(['ad_pupil', 'hr', 'rsa', 'rsp'])
    id_vars = ['participant', 'trial']
    rpca_sigs = ['pupil', 'ad_pupil']

    # PCA
    n_comp = 5

    # ex var plots
    fig = plt.figure(layout='constrained', figsize=(5*len(combs), 5*len(PERIODS)))
    subfigs = fig.subfigures(1, len(combs), wspace=0.07)
    
    all_sigs = set()
    for comb in combs:
        all_sigs = all_sigs.union(comb)
    all_data = defaultdict(dict)
    for sig in all_sigs:
        for period in PERIODS:
            print(f'importing {period} {sig:<20}', end='\r')
            clean = (sig in rpca_sigs) and rpca
            all_data[sig][period] = import_signal(sig, period, zscore, rpca=clean)

    for ind, (sigs, subfig) in enumerate(zip(combs, subfigs)):
        extra_exclude = set()
        for temp in sigs:
            extra_exclude = extra_exclude.union(FILTER[temp])

        output = []
        col = subfig.subplots(len(PERIODS), 1, sharex=True, sharey=True)
        subfig.suptitle(' '.join(sigs))
        
        # radar plots
        radar = plt.figure(layout='constrained', figsize=(9*n_comp, 6*len(PERIODS)))
        radar.suptitle(f'PCA Loadings: Top {n_comp} PC')
        sub_radars = radar.subfigures(len(PERIODS), 1, wspace=0.07)
        if zscore:
            print(f'Z-SCORED')
        print(f'Signal Combination: {sigs}')

        for ax, sub_radar, period in zip(col, sub_radars, PERIODS):
            ax.set_title(period)
            data = None
            prev = 0
            comp_map: Dict[str, Tuple[int,int]] = {}
            for sig in sigs:
                new_cols = {f'data_{x}': f'{sig}_{x}' for x in range(PERIODS[period] * SIGNALS[sig])}
                sig_data = all_data[sig][period]
                sig_data = sig_data.select(id_vars+[f'data_{x}' for x in range(PERIODS[period] * SIGNALS[sig])]).rename(new_cols)
                
                end = (PERIODS[period] * SIGNALS[sig]) - 1
                comp_map[sig] = (prev, prev+end)
                prev = end+1

                if data is None:
                    data = sig_data
                else:
                    data = data.join(sig_data, on=['participant', 'trial'], how='left')
            assert data is not None

            subs = data['participant']
            trials = data['trial']

            data = data.drop('participant', 'trial')
            # Backwards / Forwards fills -> Steamroller
            data = data.fill_null(strategy='backward')
            data = data.fill_null(strategy='forward')

            # impute mean of column
            # data = fill_null(data, period, sig)

            # Normalize
            data = data.with_columns([(pl.col(x) - pl.col(x).min())/(pl.col(x).max()-pl.col(x).min()) for x in data.columns])
            pca = PCA(data.to_numpy(), n_comp)
            red = pca.fit_transform()
            red = pl.from_numpy(np.array(red))
            red = red.with_columns([subs, trials, pl.lit(period).alias('period')])
            output.append(red)

            components = pl.from_numpy(np.array(pca.loadings))
            temp_comp = []
            for sig, (start, end) in comp_map.items():
                sig_data = components.select([f'column_{x}' for x in range(start, end+1)])
                avg = sig_data.mean(axis=1).rename(sig)
                temp_comp.append(avg)
            components = pl.DataFrame(temp_comp)
            print(components)

            ex_var = np.array(pca.ex_var) / 100
            print(ex_var)
            plot_radar(components, period, sub_radar, ex_var[:5], sigs, True)

            print(f'First 5 {period}: {ex_var[:5].sum()*100:.2f}%')
            comps = list(range(len(ex_var)))
            ax.plot(comps, ex_var)
        output = pl.concat(output).sort('participant', 'trial', 'period')
        title = f'{sigs[0]}_combs'
        (pca_out / qual[1:]).mkdir(exist_ok=True, parents=True)
        radar.savefig(str(pca_out / qual[1:] / f'{title}_loadings.png'))
        output.write_csv(str(pca_out / qual[1:] / f'c{ind}.csv'))
    fig.savefig(str(pca_out / qual[1:] / f'combs{qual}_explained_variance.png'))


def poly_test(d: int=3):
    sigs = ['pupil', 'ad_pupil']

    fig = plt.figure(layout='constrained', figsize=(5*len(sigs), 5*len(PERIODS)))
    subfigs = fig.subfigures(1, 2, wspace=0.07)
    
    for subfig, sig in zip(subfigs, sigs):
        row = subfig.subplots(len(PERIODS), 1, sharey=True)
        subfig.suptitle(sig)
        for ax, period in zip(row, PERIODS):
            time = np.array(list(range(PERIODS[period]*SIGNALS[sig]))).reshape(-1, 1)
            ax.set_title(period)
            data = import_signal(sig, period)
            # impute mean of column
            data = data.select([f'data_{x}' for x in range(PERIODS[period]*SIGNALS[sig])])
            data = data.fill_null(strategy='backward')
            data = data.to_numpy()[:1]

            for sample in data:
                model = make_pipeline(PolynomialFeatures(d), Ridge(alpha=1e-3))
                print(f'x_train: {time.shape}')
                print(time)
                print(f'y_train: {sample.shape}')
                model.fit(time, sample)
                print(model.get_params())
                y_plot = model.predict(time)
                ax.scatter(time.flatten(), sample, label="training points")
                ax.plot(time.flatten(), y_plot)
    fig.savefig('polyfit.png')


def rsa_means():
    output = []
    pca_out = Path('pca')
    for period in PERIODS:
        data = import_signal('rsa', period)
        subs = data['participant']
        trials = data['trial']

        mean = data.mean(axis=1).alias('mean')
        temp = pl.DataFrame()
        data = temp.with_columns([subs, trials, pl.lit(period).alias('period')], mean)
        print(data)
        output.append(data)
    pl.concat(output).write_csv(str(pca_out / f'rsa.csv'))


def plot_indiv_dist():
    target = Path('./pca/individual')
    groups = ['net_gain', 'win/loss_2', 'deck']

    for sig in target.iterdir():
        for method in sig.glob('*.csv'):
            for group in groups:
                method_name = method.stem.split('_')[-1]

                data = pl.read_csv(method, dtypes={'participant': pl.Int32, 'trial': pl.Int32})
                trial_data = import_trial_data()
                data = data.join(trial_data, on=['participant', 'trial'], how='left')
                value_vars = [f'column_{x}' for x in range(sum(['column' in x for x in data.columns]))]
                id_vars = list(set(data.columns).difference(value_vars))
                data = data.melt(
                    id_vars=id_vars,
                    value_vars=value_vars, 
                    variable_name='PC', 
                    value_name='value'
                )
                g = plot_dimred_dist(data, group, f'{sig.stem} {method_name}')
                g.fig.savefig(str(sig / f'{method.stem}_{group.replace("/","_")}.png'))
                plt.close(g.fig)


def plot_dimred_dist(data: pl.DataFrame, group: str, sig: str):
    split = ['net_gain', 'win/loss_2']
    g = sns.catplot(
        data=data.to_pandas(), x='quintile', 
        hue=group, y="value", kind='violin', 
        col='PC', row='period', split=group in split
    )

    g.fig.suptitle(sig)
    return g


def plot_dimred_dist_bulk():
    combinations = ['pupil', 'ad_pupil']
    remaining = ['hr', 'rsa', 'rsp']
    for subset in Path('./pca').iterdir():
        if subset.is_dir():
            for pupil, comb in zip(combinations, sorted(subset.glob('*.csv'))):
                data = pl.read_csv(comb, dtypes={'participant': pl.Int32, 'trial': pl.Int32})
                trial_data = import_trial_data()
                data = data.join(trial_data, on=['participant', 'trial'], how='left')
                value_vars = [f'column_{x}' for x in range(sum(['column' in x for x in data.columns]))]
                id_vars = list(set(data.columns).difference(value_vars))
                data = data.melt(
                    id_vars=id_vars,
                    value_vars=value_vars, 
                    variable_name='PC', 
                    value_name='value'
                )
                split = ['net_gain', 'win/loss_2']
                names = ['deck', 'gain', 'learning']
                hues = ['deck', 'net_gain', 'win/loss_2']

                for hue, name in zip(hues,names):
                    g = sns.catplot(
                        data=data.to_pandas(), x='quintile', 
                        hue=hue, y="value", kind='violin', 
                        col='PC', row='period', split=hue in split
                    )

                    g.fig.suptitle(f'Signals: {[pupil]+remaining}\nMethods: {subset.stem}')
                    g.fig.savefig(str(subset / f'{comb.stem}_{name}.png'))
                    plt.close(g.fig)


def compile_plots():
    comb = ['Raw Pupil', 'Pupil Change Scores']
    plots = list(sorted(Path('./pca/both').glob('c[0-9]*.png')))
    names = set([x.name for x in plots])
    bruh = defaultdict(list)
    for x in names:
        if 'c1' in x:
            bruh[comb[1]].append(x)
        else:
            bruh[comb[0]].append(x)
    for x in bruh.keys():
        bruh[x] = sorted(bruh[x])
    bruh = dict(bruh)
    paths = [Path('./pca/both'), Path('./pca/rpca')]

    for plot in range(len(plots) // 2):
        fig = plt.figure(layout='constrained', figsize=(35, 10))
        subfigs = fig.subfigures(1,2)
        for path, title, subfig in zip(paths,['zscored', 'normal'], subfigs):
            subfig.suptitle(title)
            axes = subfig.subplots(2,1)
            axes[0].axis('off')
            axes[1].axis('off')
            axes[0].imshow(Image.open(path / bruh[comb[0]][plot]))
            axes[1].imshow(Image.open(path / bruh[comb[1]][plot]))
            axes[0].set_title(comb[0])
            axes[1].set_title(comb[1])
        fig.savefig(str(Path('./pca') / bruh[comb[0]][plot]))

    print(bruh)


def performance():
    P = 10
    iterations = 500
    if not Path(f'score_sim_{P}.csv').exists():
        data = import_trial_data().drop('win/loss', 'win/loss_2', 've_median_split', 'valid_score')
        probs = group_monte_carlo(data, 'trial', P, 'participant')
        simulations = []

        for sub, probs in probs.items():
            print(f'Simulating subject {sub}')
            group_sim = simulate(sub, generate_selections, gen_incorrect, iterations, probs)

            # CHANGES
            group_sim = group_sim.select('iter', 'trial', 'score')
            group_sim = group_sim.pivot(values="score", index="iter", columns="trial")
            group_sim = group_sim.with_columns(
                [
                    pl.lit(sub).alias('participant'),
                ]
            )
            simulations.append(group_sim)

        simulations = pl.concat(simulations).groupby('participant').mean().drop('iter').sort('participant')
        simulations.write_csv(f'score_sim_{P}.csv')
    else:
        simulations = pl.read_csv(f'score_sim_{P}.csv')
    # grouping(simulations)
    test(simulations)


def test(data):
    
    org_data = import_trial_data().drop('win/loss', 'win/loss_2', 've_median_split', 'valid_score')

    trial_loss_win_count = org_data.select('participant', 'net_gain').groupby('participant', 'net_gain').count()
    trial_loss_win_count = trial_loss_win_count.pivot('count', 'participant', 'net_gain').with_columns((pl.col('gain')-pl.col('loss')).alias('magnitude'))
    org_data = org_data.pivot(values="score", index=['participant'], columns="trial")
    print(trial_loss_win_count)

    # sns.stripplot(data.to_pandas(), x='99')

    subs = org_data.drop_in_place('participant')
    cleaned = org_data.transpose(column_names=subs.cast(pl.Utf8))
    output = defaultdict(list)
    for sub, column in zip(subs, cleaned):
        sub_score = cleaned.select(column).to_series()
        output['participant'].append(sub)
        output['arg_min'].append(sub_score.arg_min())
        output['arg_max'].append(sub_score.arg_max())
        output['max'].append(sub_score.max())
        output['min'].append(sub_score.min())
        output['amplitude'].append(np.abs(sub_score.min() - sub_score.max()))


    cleaned = pl.DataFrame(output)
    print(cleaned)
    fig, axes = plt.subplots(2,3, figsize=(20,10))

    sns.histplot(cleaned.to_pandas(), x='amplitude', ax=axes[0][0])
    sns.histplot(cleaned.to_pandas(), x='arg_min', ax=axes[0][1])
    sns.histplot(cleaned.to_pandas(), x='arg_max', ax=axes[0][2])
    sns.histplot(cleaned.to_pandas(), x='min', ax=axes[1][0])
    sns.histplot(cleaned.to_pandas(), x='max', ax=axes[1][1])
    sns.histplot(trial_loss_win_count.to_pandas(), x='magnitude', ax=axes[1][2])

    # sns.heatmap(cleaned.to_pandas().cov(), robust=True)
    # sns.histplot(data=data, x="99", binwidth=200)
    # sns.histplot(data=org_data.to_pandas(), x="99", binwidth=200)

    plt.show()


def grouping(data: pl.DataFrame=pl.DataFrame()):
    if data.is_empty():
        data = import_trial_data().drop('win/loss', 'win/loss_2', 've_median_split', 'valid_score')
        data = data.pivot(values="score", index=['participant'], columns="trial")
    
    pca = PCA(data.to_numpy(), 2).fit_transform() # type: ignore
    tsne = TSNE().fit_transform(data.to_numpy())
    umap = UMAP().fit_transform(data.to_numpy())
    umap_dens = UMAP(densmap=True).fit_transform(data.to_numpy())

    dim_red = pd.DataFrame(pca, columns=['x','y'])
    dim_tsne = pd.DataFrame(tsne, columns=['x','y'])
    umap = pd.DataFrame(umap, columns=['x','y'])
    umap_dens = pd.DataFrame(umap_dens, columns=['x','y'])

    fig, ax = plt.subplots(2,2, figsize=(20,20))

    sns.scatterplot(dim_red, x='x', y='y', ax=ax[0][0])
    sns.scatterplot(dim_tsne, x='x', y='y', ax=ax[0][1])

    sns.scatterplot(umap, x='x', y='y', ax=ax[1][0])
    sns.scatterplot(umap_dens, x='x', y='y', ax=ax[1][1])

    fig.savefig('test.png')


def main() -> int:
    # dim_red_multisig(rpca=True)
    # dim_red_multisig(zscore=True, rpca=True)
    # plot_dimred_dist_bulk()
    # compile_plots()
    # dimred_sig(True, True)
    # plot_indiv_dist()
    # grouping()
    performance()
    return 0

if __name__ == '__main__':
    sys.exit(main())