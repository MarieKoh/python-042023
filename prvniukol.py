import pandas
presidents = pandas.read_csv("1976-2020-president.csv")
presidents['candidate_rank'] = presidents.groupby(['year', 'state', 'party_simplified'])['candidatevotes'].rank(ascending=False)
print(presidents.head())
presidents = presidents[presidents['candidate_rank'] == 1]
print(presidents.head())
print(presidents.columns)

import numpy as np
presidents['previous_winner'] = presidents.groupby('state')['party_simplified'].shift()
presidents['winning_party_change'] = np.where((presidents['year'] != 1976) & (presidents['party_simplified'] != presidents['previous_winner']), True, False)
changes = presidents.loc[presidents['winning_party_change'] == True, ['state', 'year']]
changes = changes.groupby('state').count().reset_index()
changes = changes.sort_values(by='year', ascending=False)
print(changes[['state', 'year']].head(15))

presidents["change"] = presidents["party_simplified"] != presidents["previous_winner"]
data_pivot = presidents.groupby("state")["change"].sum()
data_pivot = pandas.DataFrame(data_pivot)
data_pivot = data_pivot.sort_values("change", ascending=False)
print(data_pivot)

import matplotlib.pyplot as plt

plt.bar(data_pivot.index[:10], data_pivot["change"][:10])
plt.xticks(rotation=90)
plt.xlabel("Stát")
plt.ylabel("Počet změn strany")
plt.title("Státy s nějvětšími změnami ve stranách v prezidentských volbách")
plt.show()

firsttwo = presidents.groupby(['year', 'state']).head(2)
print(firsttwo.head())

firsttwo.loc[:, 'abs_margin'] = abs(firsttwo['candidatevotes'] - firsttwo.groupby(['year', 'state'])['candidatevotes'].shift(-1))
firsttwo.loc[:, 'rel_margin'] = firsttwo['abs_margin'] / firsttwo['totalvotes']
min_margin = firsttwo.groupby('year')['rel_margin'].min()
for year, margin in min_margin.items():
    state = firsttwo.loc[(firsttwo['year'] == year) & (firsttwo['rel_margin'] == margin), 'state'].values[0]
    print(f"Nejtěsnější výsledek v roce {year} byl v {state} s relativním marginem {margin}")

pivot_table = pandas.pivot_table(firsttwo, values='rel_margin', index=['state', 'year'], aggfunc=np.min)
pivot_table.reset_index(inplace=True)
pivot_table.columns = [''] + list(pivot_table.columns[1:])
print(pivot_table)
