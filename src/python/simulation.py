from random import sample, randint, shuffle, choices
from collections import defaultdict
from typing import Callable, Union, Dict
import sys
import polars as pl
from pathlib import Path
import os
import numpy as np
from utils import *
from numpy.random import choice
import matplotlib.pyplot as plt
from shutil import rmtree

dir_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_path)
PILES = ['DeckA', 'DeckB', 'DeckC', 'DeckD']
OUTPUT = Path('./simulations')
DEBUG = False

MonteCarloSeries = Dict[Tuple[int, int], np.ndarray]
TrialProbs = Tuple[str, Tuple[List[int], List[int]]]
N = 100
TIME = np.arange(N)

def monte_carlo(data: pl.DataFrame, time_col: str, p: int) -> MonteCarloSeries:
    min_t, max_t = data[time_col].min(), data[time_col].max()
    quantiles = [(x[0],x[-1]) for x in np.array_split(np.arange(min_t, max_t+1), p)] # type: ignore
    output = {}
    
    for ind, quantile in enumerate(quantiles):
        start, end = quantile
        if not ind:
            start += 1
        quantile_data = data.filter((pl.col(time_col) >= start) & (pl.col(time_col) <= end))
        probs = TransitionMatrix(quantile_data, PILES).to_numpy()
        output[(start, end)] = probs
    return output


def group_monte_carlo(data: pl.DataFrame, time_col: str, p: int, group: str) -> Dict[str, MonteCarloSeries]:
    if data[group].dtype is pl.Utf8:
        data = data.filter((pl.col(group) != '') & (pl.col(group).is_not_null()))
    members = data[group].unique()

    matricies = {}

    for member in members:
        sub_group = data.filter(pl.col(group) == member)
        matricies[member] = monte_carlo(sub_group, time_col, p)
    return matricies


def generate_selections(matricies: Dict[Tuple[int, int], np.ndarray]) -> np.ndarray:
    probs = sorted(matricies.items())
    selections = []

    # Choose first deck at random
    temp = probs[0][1]
    first_choice = 0
    for ind, deck in enumerate(temp):
        if sum(deck) !=0:
            first_choice = ind
            break

    selections.append(choice(PILES, p=temp[first_choice]))

    for bounds, prob in probs:
        start, end = bounds
        for x in range(start, end+1):
            bruh = PILES.index(selections[x-1])
            if prob[bruh].sum() != 0:
                selections.append(choice(PILES, p=prob[bruh]))
            else:
                for x in prob:
                    if x.sum() != 0:
                        selections.append(choice(PILES, p=x))
                        break
    
    return np.array(selections)


def random_select() -> np.ndarray:
    return np.array(choices(PILES, k=100))


def reset() -> Tuple[List[int], List[int]]:
    a_loss = [150, 200, 250, 300, 350]
    c_loss = [0, 25, 50, 50, 75]
    shuffle(a_loss)
    shuffle(c_loss)
    return a_loss, c_loss    


def gen_correct() -> TrialProbs:
    low_fails: List[int] = []
    high_fails: List[int] = []
    for i in range(10):
        domain = sample(range(0, 9), 5)
        for j in domain:
            high_fails += [i * 10 + j]
        low_fails += [i * 10 + sample(list(set(range(0,9)).difference(domain)), 1)[0]]
    sorted(low_fails)
    sorted(high_fails)

    if DEBUG:
        print(f'high fails: \ncount: {len(high_fails)}\n{high_fails}')
        print(f'low fails: \ncount: {len(low_fails)}\n{low_fails}')
        print(f'intersection: \n{set(high_fails).intersection(low_fails)}')
    return 'correct', (low_fails, high_fails)


def gen_incorrect() -> TrialProbs:
    low_fails: List[int] = []
    high_fails: List[int] = []
    for i in range(10):
        for j in sample(range(0, 9), 5):
            high_fails += [i * 10 + j]
        low_fails += [i * 10 + randint(0, 9)]
    sorted(low_fails)
    sorted(high_fails)
    
    if DEBUG:
        print(f'high fails: \ncount: {len(high_fails)}\n{high_fails}')
        print(f'low fails: \ncount: {len(low_fails)}\n{low_fails}')
        print(f'intersection: \n{set(high_fails).intersection(low_fails)}')
    return 'incorrect', (low_fails, high_fails)


def simulate(group_name: str, method: Callable, trial_prob_func: Callable, n: int, args: Union[None,MonteCarloSeries] = None):
    data = defaultdict(list)
    fail = ''
    for x in range(n):
        fail, (low_fails, high_fails) = trial_prob_func()

        a_loss, c_loss = reset()
        b_loss = 1250
        d_loss = 250
        
        outcomes: Dict[str, Tuple[str,int]] = {
            'DeckA': ('hghf',100),
            'DeckB': ('hglf',100), 
            'DeckC': ('lghf',50), 
            'DeckD': ('lglf',50)
        }
        outcome: str = ""
        total = 2000

        if args is None:
            selections: np.ndarray = method()
        else:
            selections: np.ndarray = method(args)

        for trial, stimulus in zip(np.arange(100), selections):
            gain: int = 0
            loss: int = 0

            if trial % 10:
                a_loss, c_loss = reset()

            outcome, gain = outcomes[stimulus]

            if outcome != "":
                if "lg" in outcome:
                    if "lf" in outcome and trial in low_fails:
                        loss = d_loss
                    elif "hf" in outcome and trial in high_fails:
                        loss = c_loss.pop(0)
                elif "hg" in outcome:
                    if "lf" in outcome and trial in low_fails:
                        loss = b_loss
                    elif "hf" in outcome and trial in high_fails:
                        loss = a_loss.pop(0)

            total += gain - loss
            data['iter'].append(x)
            data['trial'].append(trial)
            data['deck'].append(stimulus)
            data['gain'].append(gain)
            data['loss'].append(loss)
            data['score'].append(total)
    
    data = pl.DataFrame(data).with_columns(
        [
            pl.lit(group_name).alias('subgroup'),
            pl.lit(fail).alias('fail')
        ]
    )
    return data


def get_unique(data: pl.DataFrame, name: str):
    out_dir = OUTPUT / 'unique'
    out_dir.mkdir(exist_ok=True, parents=True)

    data = data.select('deck', 'gain', 'loss')
    output = data.groupby('deck', 'gain', 'loss').count()
    sums = output.groupby('deck').sum()

    new_output = []
    for deck in output['deck'].unique():
        count = sums.filter(pl.col('deck') == deck).to_dicts()[0]['count']
        new_output.append(output.filter(pl.col('deck') == deck).with_columns([(pl.col('count') / count).alias('percent')]))
    output = pl.concat(new_output).sort('deck', 'gain', 'loss')
    output.write_csv(str(out_dir / f'{name}.csv'))


def fast_group_plot(simulations: pl.DataFrame, iterations: int, p: int):
    STD_ERR = np.sqrt(iterations)

    output = Path(f'./simulations/plots/scores')
    if output.exists():
        rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    data_cols = [f'{x}' for x in range(100)]
    id_vars = 'group', 'subgroup', 'Trial Risk Method'
    avgs = simulations.groupby(id_vars).agg(
        [pl.col(f'{x}').mean() for x in range(100)]
    )
    np_avgs = avgs.select(data_cols).to_numpy()
    stds = simulations.groupby(id_vars).agg(
        [pl.col(f'{x}').std() for x in range(100)]
    )
    np_stds = stds.select(data_cols).to_numpy()

    groups = avgs.select(id_vars).to_numpy()
    u_groups = set(list(groups[:,0]))

    fig, axes = plt.subplots(1,len(u_groups), figsize=(len(u_groups)*5, 5), sharex=True, sharey=True)
    fig.suptitle(f'Simulation: N={iterations} | P={p}')
    axes = dict(zip(u_groups, axes))

    colors = plt.rcParams["axes.prop_cycle"]()
    sampled_colors = defaultdict(dict)
    for g, sg in zip(groups[:,0], groups[:,1]):
        sampled_colors[g][sg] = next(colors)["color"]

    for group, subgroup, fail, avg, std in zip(groups[:,0], groups[:,1], groups[:,2], np_avgs, np_stds):
        ci_u = []
        ci_l = []

        for x in range(len(std)):
            sem = (0.95 * (std[x]/STD_ERR))
            ci_u.append(avg[x] + sem)
            ci_l.append(avg[x] - sem)

        ci_u = np.array(ci_u)
        ci_l = np.array(ci_l)

        ax = axes[group]
        c = sampled_colors[group][subgroup]
        if fail == 'correct':
            ax.plot(TIME, avg, label=f'{subgroup.capitalize()} - {fail}', color=c)    
        else:
            ax.plot(TIME, avg, label=f'{subgroup.capitalize()} - {fail}', linestyle='dashed', color=c)
        ax.fill_between(TIME, ci_l, ci_u, color=c, alpha=.1) # type: ignore

        ax.set_title(f'{group}')
    [ax.legend() for ax in axes.values()]
        
    fig.tight_layout()
    fig.savefig(str(output / f'plot_{iterations}_{p}.png'))
    plt.close(fig)


def generate_sim_plots(iterations:int=5000):
    P = 20
    simulations = []
    
    failure = {
        'correct': gen_correct,
        'incorrect': gen_incorrect
    }
    for key, val in failure.items():
        # Simulate random selection of piles
        print(f'Simulating Random')
        rand = simulate('random', random_select, val, iterations)

        # CHANGES
        rand = rand.select('iter', 'trial', 'score')
        rand = rand.pivot(values="score", index="iter", columns="trial")

        rand = rand.with_columns(
            [
                pl.lit('random').alias('group'),
                pl.lit('random').alias('subgroup'),
                pl.lit(key).alias('Trial Risk Method')
            ]
        )
        simulations.append(rand)

        # Simulate with actual data
        data = import_trial_data()
        groups = ['win/loss_2', 've_median_split']

        for group_col in groups:
            matricies = group_monte_carlo(data, 'trial', P, group_col)

            for group, probs in matricies.items():
                print(f'Simulating {group.capitalize()}')
                group_sim = simulate(group, generate_selections, val, iterations, probs)

                # CHANGES
                group_sim = group_sim.select('iter', 'trial', 'score')
                group_sim = group_sim.pivot(values="score", index="iter", columns="trial")

                group_sim = group_sim.with_columns(
                    [
                        pl.lit(group_col).alias('group'),
                        pl.lit(group).alias('subgroup'),
                        pl.lit(key).alias('Trial Risk Method')
                    ]
                )
                simulations.append(group_sim)
        
    simulations = pl.concat(simulations)
    fast_group_plot(simulations, iterations, P)


def show_err():
    gen_correct()
    gen_incorrect()


def main() -> int:
    # generate_sim_plots()
    show_err()
    return 0


if __name__ == '__main__':
    sys.exit(main())