import jax.numpy as jnp
from JaxLDA import LDA
from JaxPCA import PCA
import pandas as pd
from pathlib import Path
from utils import *
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
import json
import dtreeviz
from shapash import SmartExplainer
from numpy import random
import numpy.ma as ma
from PIL import Image
import io
from cairosvg import svg2png
import os

def visualize_model(modelname: str, model: lgb.Booster, X_train, y_train, n_classes, features, output):
    output_path = Path(f'./model_viz/trees/{modelname}')
    output_path.mkdir(exist_ok=True, parents=True)

    for x in random.choice(np.arange(model.num_trees()), 5, replace=False):
        viz_model = dtreeviz.model(model, tree_index=x,
            X_train=X_train, y_train=y_train,
            feature_names=features,
            target_name=output, class_names=n_classes
        )
        bruh = viz_model.view().save_svg()

        with open(bruh) as fp:
            svg2png(bytestring=fp.read(),write_to=str(output_path / f'tree_{x}.png'))
        os.remove(bruh)


def plot_feat_contrib(xpl: SmartExplainer, features: List[str], name: str, output: Path):
    fig, axes = plt.subplots(3,4, figsize=(20,15))
    axes = np.array(axes).flatten()
    for ax, feature in zip(axes, features):
        image_data = xpl.plot.contribution_plot(col=feature).to_image(format="png")
        ax.imshow(Image.open(io.BytesIO(image_data)))
    fig.tight_layout()
    fig.savefig(str(output / f'{name}_feat_cont.png'))


def shapash_bruh(model: lgb.LGBMClassifier, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                 y_train: pd.Series, y_test: pd.Series, y_pred: pd.Series, name: str):
    output_path = Path(f'./model_viz/plots')
    plot_types = ['compacity', 'feat_imp', 'neighbors', 'feat_contrib', 'stability']
    for x in plot_types:
        (output_path / x).mkdir(exist_ok=True, parents=True)

    xpl = SmartExplainer(
        model=model,
        # features_dict=house_dict,  # Optional parameter
        # preprocessing=encoder, # Optional: compile step can use inverse_transform method
        # postprocessing=postprocess, # Optional: see tutorial postprocessing  
    )

    xpl.compile(
        x=X_test,    
        y_pred=y_pred, # Optional: for your own prediction (by default: model.predict)
        y_target=y_test, # Optional: allows to display True Values vs Predicted Values
    )

    if 'deck' not in name:
        xpl.plot.compacity_plot().write_image(str(output_path / plot_types[0] /  f"{name}_compact.png"))
        rand_ex = random.randint(0, y_pred.size)
        xpl.plot.local_neighbors_plot(index=rand_ex).write_image(str(output_path / plot_types[2] / f"{name}_neigh.png"))
        rand_sel = random.choice(np.arange(y_pred.size), 20, replace=False).tolist()
        xpl.plot.stability_plot(selection=rand_sel, distribution="boxplot").write_image(str(output_path / plot_types[4] / f"{name}_stab.png"))

    xpl.plot.features_importance().write_image(str(output_path / plot_types[1] / f"{name}_feat.png"))
    plot_feat_contrib(xpl, X_test.columns.to_list(), name, output_path / plot_types[3])


def eval_model(X_train, X_test, y_train, y_test, opt, n_classes, name: str) -> Tuple[float, lgb.LGBMClassifier, Dict]:
    if 'acc' in opt:
        del opt['acc']
    Path('models').mkdir(exist_ok=True, parents=True)
    gbm = lgb.LGBMClassifier(**opt, gpu_platform_id = 1, gpu_device_id = 0)
    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(5)])
    
    gbm.booster_.save_model(f'models/{name}')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

    def accuracy(y_test, y_pred) -> float:
        return np.sum(y_test==y_pred) / y_test.size

    def class_acc(y_test, y_pred):
        output = {}
        for unq_y in n_classes:
            sample_test_y = ma.array(y_test, mask = y_test==unq_y).compressed()
            sample_pred_y = ma.array(y_pred, mask = y_test==unq_y).compressed()
            if isinstance(unq_y, float):
                label = int(unq_y)
            else:
                label = str(unq_y)
            output[label]=np.sum(sample_test_y==sample_pred_y) / sample_test_y.size
        return output

    acc = accuracy(y_test, y_pred)
    class_out = class_acc(y_test, y_pred)
    print(f'The accuracy of prediction is: {acc}')
    return acc, gbm, class_out


def get_opt_hyper(X_train, y_train, n_classes):
    estimator = lgb.LGBMClassifier(gpu_platform_id = 1, gpu_device_id = 0)#,num_leaves=n_classes.size)
    leaves = np.arange(4) + 1
    leaves = leaves * n_classes.size

    param_grid = {
        'boosting_type': ['gbdt', 'dart'],
        'learning_rate': [0.01, 0.1, 1.0],
        'n_estimators': [20, 40, 60, 120, 200],
        'num_leaves': leaves.tolist(),
        'min_data_in_leaf': (np.array([0.2,0.4,0.6]) * X_train.shape[0]).astype(int).tolist()
    }

    gbm = GridSearchCV(estimator, param_grid, cv=5) # type: ignore
    gbm.fit(X_train, y_train)
    opt = gbm.best_params_
    return opt


def run_lgb(X_train, X_test, y_train, y_test, name: str):
    n_classes = np.unique(np.concatenate((y_train, y_test)))
    opt = get_opt_hyper(X_train, y_train, n_classes)
    opt['acc'], _, _ = eval_model(X_train, X_test, y_train, y_test, opt, n_classes, name)
    return opt


def load_opt(target: str = 'models.json'):
    opt = {}
    if (Path(f'./{target}').exists()):
        with open(Path(f'./{target}'), 'r') as fp:
            opt = json.load(fp)
    return opt


def quick_analyze(zscore: bool = False):
    option = ''
    out_data = Path('./out_data_pca')
    out_data.mkdir(exist_ok=True, parents=True)
    if zscore:
        option = '_zscore'

    combs = []
    # baseline - 360, 1000, 25, 1000
    combs.append(['pupil', 'hr', 'rsa', 'rsp'])
    combs.append(['ad_pupil', 'hr', 'rsa', 'rsp'])
    n_methods = 3

    factors= ['win/loss_2', 'net_gain', 'deck']
    opt = load_opt('models2.json')
    if opt is not None:
        out_opt = opt
    else:
        out_opt = {}

    all_class_acc = defaultdict(dict)

    for factor in factors:
        factor_name = factor.replace("/","_")
        if factor not in out_opt:
            out_opt[factor] = {}
        output = Path('./supervised') / factor_name

        for ind, sigs in enumerate(combs):
            # skip pupil
            if ind == 0:
                continue
            if f'c{ind}' not in out_opt[factor]:
                out_opt[factor][f'c{ind}']={}
            extra_exclude = set() 
            for temp in sigs:
                extra_exclude = extra_exclude.union(FILTER[temp])
            
            fig = plt.figure(layout='constrained', figsize=(10*n_methods, 10*len(PERIODS)))
            subfigs = fig.subfigures(1,len(PERIODS))
            for subfig, period in zip(subfigs, PERIODS):
                subfig.suptitle(period)
                data = pl.read_csv(str(out_data / f'c{ind}_{period}{option}.csv'))

                if factor == 'net_gain':
                    losses = data.filter(pl.col("net_gain")=='loss')
                    wins = data.filter(pl.col("net_gain")=='gain').sample(losses.height, shuffle=True)
                    data = pl.concat([losses, wins])

                y = data[factor].to_numpy()
                classes = np.unique(y)
                label_map = dict(zip(np.arange(len(classes)), classes))
                y_float = np.zeros(y.size)
                for label, org in label_map.items():
                    y_float[y==org] = label
                y_float = y_float.astype(int)
                data = data.drop(['participant', 'trial']+factors)
                # Backwards / Forwards fills -> Steamroller
                data = data.fill_null(strategy='backward')
                data = data.fill_null(strategy='forward')
                X = data.to_numpy()                

                X_train, X_test, y_train, y_test = train_test_split(X, y_float, test_size=0.33, random_state=42)

                if period not in out_opt[factor][f'c{ind}']:
                    opt = run_lgb(X_train, X_test, y_train, y_test, f'{factor_name}_c{ind}_{period}.txt')
                    out_opt[factor][f'c{ind}'][period] = opt
                else:
                    indiv_opt = opt[factor][f'c{ind}'][period]
                    n_classes = np.unique(np.concatenate((y_train, y_test)))
                    acc, gbm, class_acc = eval_model(
                        X_train, X_test, y_train, 
                        y_test, indiv_opt, n_classes, f'{factor_name}_c{ind}_{period}.txt'
                    ) # type: ignore
                    all_class_acc[factor][period] = class_acc
                    name = f'c{ind}_{period}_{factor_name}'
                    temp1 = pl.from_numpy(X_train, schema=data.columns).with_columns(pl.Series(factor, y_train)).to_pandas()
                    temp2 = pl.from_numpy(X_test, schema=data.columns).with_columns(pl.Series(factor, y_test)).to_pandas()
                    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
                    shapash_bruh(
                        gbm,
                        temp1[data.columns], 
                        temp2[data.columns],
                        temp1[factor], 
                        temp2[factor], 
                        pd.Series(y_pred, name=factor), name
                    )
                    visualize_model(name, gbm.booster_, temp1[data.columns], temp1[factor], n_classes, data.columns, factor)
                    opt[factor][f'c{ind}'][period]['acc'] = acc

    with open("models2.json","w") as fp:
        json.dump(out_opt, fp, indent=4, sort_keys=True)
    with open('trait_models_acc.json', 'w+') as fp:
        json.dump(all_class_acc, fp, indent=1, sort_keys=True)


def plot_acc():
    data = pl.read_csv('./models.csv').rename({'accuracy': 'accuracy %'}).filter(pl.col('combination') != 'c0')
    data = data.with_columns(pl.col('accuracy %') * 100)
    bruh = { 'win/loss_2': 'learners/non-learners'}
    data = data.with_columns(pl.col('feature').map_dict(bruh, default=pl.first()))
    sns.set(font_scale=1.9)
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    sns.barplot(data.to_pandas(), x='feature', y='accuracy %', hue='period')

    fig.subplots_adjust(top=.9)
    fig.suptitle('Predicted Performance Outcomes by Period', fontsize = 24)
    ax.axhline(25, linestyle='--', color='black')
    ax.text(0,26, "Deck Chance")

    ax.axhline(50, linestyle='--', color='black')
    ax.text(0, 47, "Net Change/Learners Chance")
    fig.savefig(f'model_acc.png')


def compile_feat_json(target: str = "trait_models_acc.json"):
    with open(target,"r") as fp:
        summary = json.load(fp)
    print(summary)

    data = defaultdict(list)
    for feature, feat_inst in summary.items():
        for period, all_classes in feat_inst.items():
            for c, acc in all_classes.items():
                data['feature'].append(feature)
                data['class'].append(c)
                data['period'].append(period)
                data['accuracy'].append(acc)
    data = pl.DataFrame(data)
    data.write_csv(f'class_acc.csv')


def plot_feat_acc():
    data = pl.read_csv('class_acc.csv').rename({'accuracy': 'accuracy %'})
    data = data.with_columns(pl.col('accuracy %') * 100)
    bruh = { 'win/loss_2': 'learners/non-learners'}
    bruh2 = {'winner': 'learner', 'loser': 'non-learner'}
    data = data.with_columns(
        [
            pl.col('feature').map_dict(bruh, default=pl.first()),
            pl.col('class').map_dict(bruh2, default=pl.first()),
        ]
    )
    

    g = sns.catplot(data.to_pandas(), x='class', y='accuracy %', hue='period', col='feature', kind='bar', sharex=False)
    g.fig.subplots_adjust(top=.9)
    g.fig.suptitle('Accuracy Within Feature Between Period by Class', fontsize = 15)
    g.fig.savefig(f'class_acc.png')


def analyze(zscore: bool = False):
    option = ''
    out_data = Path('./out_data_pca')
    out_data.mkdir(exist_ok=True, parents=True)
    if zscore:
        option = '_zscore'
    combs = []
    combs.append(['pupil', 'hr', 'rsa', 'rsp'])
    combs.append(['ad_pupil', 'hr', 'rsa', 'rsp'])
    id_vars = ['participant', 'trial']
    rpca_sigs = ['pupil', 'ad_pupil']
    n_methods = 3

    all_sigs = set()
    for comb in combs:
        all_sigs = all_sigs.union(comb)
    all_data = defaultdict(dict)
    for sig in all_sigs:
        for period in PERIODS:
            print(f'importing {period} {sig:<20}', end='\r')
            clean = sig in rpca_sigs
            all_data[sig][period] = import_signal(sig, period, zscore, rpca=clean)

    factors= ['win/loss_2', 'net_gain', 'deck']
    opt = load_opt()

    for factor in factors:
        if factor not in opt:
            opt[factor] = {}
        output = Path('./supervised') / factor.replace("/","_")

        for ind, sigs in enumerate(combs):
            if f'c{ind}' not in opt[factor]:
                opt[factor][f'c{ind}']={}
            extra_exclude = set() 
            for temp in sigs:
                extra_exclude = extra_exclude.union(FILTER[temp])
            
            fig = plt.figure(layout='constrained', figsize=(10*n_methods, 10*len(PERIODS)))
            subfigs = fig.subfigures(1,len(PERIODS))
            for subfig, period in zip(subfigs, PERIODS):
                subfig.suptitle(period)
                data = None
                prev = 0
                comp_map: Dict[str, Tuple[int,int]] = {}
                for sig in sigs:
                    new_cols = {f'data_{x}': f'{sig}_{x}' for x in range(PERIODS[period] * SIGNALS[sig])}
                    sig_data = all_data[sig][period]
                    subs = sig_data['participant']
                    trial = sig_data['trial']

                    temp_sig = sig_data.select(list(new_cols.keys()))
                    temp_sig = temp_sig.with_columns([pl.col(x).fill_null(pl.col(x).mean()) for x in temp_sig.columns])
                    temp_sig = temp_sig.with_columns(
                        [(pl.col(x) - pl.col(x).min())/(pl.col(x).max()-pl.col(x).min()) for x in temp_sig.columns]
                    )

                    pca = PCA(temp_sig.to_numpy(), 3) # type: ignore
                    red = pca.fit_transform()
                    red = pl.from_numpy(np.array(red), schema=[f'{sig}_pc_{x+1}' for x in range(3)])
                    red = red.with_columns([subs, trial])

                    comp_map[sig] = (prev, prev+2)
                    prev += 3

                    if data is None:
                        data = red
                    else:
                        data = data.join(red, on=['participant', 'trial'], how='left')
                assert data is not None

                subs = data['participant']
                trials = data['trial']
                cols = id_vars + factors
                trial_info = import_trial_data()[cols]
                data = data.join(trial_info, on=id_vars, how='left')
                y = data[factor].to_numpy()
                classes = np.unique(y)
                label_map = dict(zip(np.arange(len(classes)), classes))
                y_float = np.zeros(y.size)
                for label, org in label_map.items():
                    y_float[y==org] = label
                y_float = y_float.astype(int)

                data.write_csv(str(out_data / f'c{ind}_{period}{option}.csv'))
                data = data.drop(['participant', 'trial']+factors)
                # Backwards / Forwards fills -> Steamroller
                data = data.fill_null(strategy='backward')
                data = data.fill_null(strategy='forward')
                X = data.to_numpy()

                X_train, X_test, y_train, y_test = train_test_split(X, y_float, test_size=0.33, random_state=42)

                if period not in opt[factor][f'c{ind}']:
                    one_opt = run_lgb(
                        X_train, X_test, y_train, y_test, f'{factor}_c{ind}_{period}.txt'
                    )
                    opt[factor][f'c{ind}'][period] = one_opt
                else:
                    indiv_opt = opt[factor][f'c{ind}'][period]
                    n_classes = np.unique(np.concatenate((y_train, y_test)))
                    acc, _, _ = eval_model(
                        X_train, X_test, y_train, y_test,
                        indiv_opt, n_classes, f'{factor}_c{ind}_{period}.txt'
                    )
                    opt[factor][f'c{ind}'][period]['acc'] = acc

                # # UMAP
                # umap = UMAP().fit(X_train, y_train)
                # umap_emb = umap.transform(X_test)

                # # densMAP
                # # Cannot do this yet
                # # dens = UMAP(densmap=True).fit(X_train, y_train)
                # # dens_emb = dens.transform(X_test)
                # dens_emb = UMAP(densmap=True).fit_transform(X, y_float)
                
                # # Normalize
                # data = data.with_columns([(pl.col(x) - pl.col(x).min())/(pl.col(x).max()-pl.col(x).min()) for x in data.columns])
                # data = jnp.array(data.to_numpy())
                # model = LDA(data, y, 2)
                # dim_red = model.fit()

                # (output / period).mkdir(exist_ok=True, parents=True)

                # ex_var = model.plot_ex_var()
                # ex_var.savefig(str(output / period / f'exvar_lda_{ind}.png'))

                # embeddings = {'UMAP': (umap_emb, y_test), 'densMAP': (dens_emb, y_float)}
                # axes = subfig.subplots(len(embeddings),1)
                # for ax, (method, (red_data, ys)) in zip(axes, embeddings.items()):
                #     red = pd.DataFrame(red_data, columns=['x','y'])
                #     red['class'] = ys
                #     ax.set_title(method)
                #     sns.scatterplot(red, x='y', y='x', hue='class', ax=ax)
            fig.savefig(str(output / f'embeddings_{ind}.png'))
            plt.close(fig)
    with open("models.json","w") as fp:
        fp.write(json.dumps(opt, indent=4, sort_keys=True))


def compile_json(target: str = "models.json"):
    with open(target,"r") as fp:
        summary = json.load(fp)
    print(summary)

    data = defaultdict(list)
    for feature, combinations in summary.items():
        for comb, periods in combinations.items():
            for period, metrics in periods.items():
                data['feature'].append(feature)
                data['period'].append(period)
                data['combination'].append(comb)
                data['accuracy'].append(metrics['acc'])
                data['learning_rate'].append(metrics['learning_rate'])
                data['n_estimators'].append(metrics['n_estimators'])
    summary = pl.DataFrame(data)
    print(summary)
    summary.write_csv('models.csv')


def main() -> int:
    #analyze(zscore=True)
    # compile_json('models2.json')
    quick_analyze(zscore=True)
    # plot_acc()
    # test()
    # compile_feat_json()
    # plot_feat_acc()

    return 0


if __name__ == '__main__':
    sys.exit(main())