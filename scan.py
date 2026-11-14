"""
Run Mizan on every built-in city, variable and season and save the results.
Uses the saved copies in data/, so it needs no internet.

Run with:  python scan.py     ->  results/all_results.csv

AI assistance: drafted with Claude (Anthropic) during the hackathon. See README.md.
"""
from pathlib import Path

import pandas as pd

import power
import stats

rows = []
for param in power.VARIABLES:
    for season, months in power.SEASONS.items():
        block = []
        for city, (lat, lon) in power.CITIES.items():
            table, meta = power.fetch(lat, lon, param)
            y, x = power.yearly(table, months)
            v = stats.judge(y, x)
            if v.trend is None: continue
            block.append(dict(param=param, season=season, city=city, verdict=v.label, note=(v.notes or [''])[0],
                              slope10=v.trend.slope*10, lo=v.trend.slope_low*10, hi=v.trend.slope_high*10,
                              p=v.trend.p, plain_p=stats.naive_p(y, x), agree=v.windows.agreement,
                              shift_year=v.step.break_year if v.step.is_step else None, units=meta['units']))
        adj, keep = stats.benjamini_hochberg([b['p'] for b in block])
        for b, a, k in zip(block, adj, keep):
            b['adj_p'] = a
            if b['verdict'] == stats.ROBUST and not k: b['verdict'] = stats.INCONCLUSIVE; b['note'] = 'many_places'
        rows += block
Path('results').mkdir(exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv('results/all_results.csv', index=False)
print(len(df), 'results')
print(df.groupby(['param','verdict']).size().unstack(fill_value=0))
