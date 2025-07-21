#%%
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

#%%
model = tree.DecisionTreeClassifier(random_state=42)

#%%
df = pd.read_parquet("data/dados_clones.parquet")
df.head()

#%%
target = 'Status '

features = ['Massa(em kilos)', 'Estatura(cm)', 'Tempo de existência(em meses)']

y = df[target]
X = df[features]

#%%
# Transformando os dados
X = X.replace({
    'Tipo 1': 1,
    'Tipo 2': 2,
    'Tipo 3': 3,
    'Tipo 4': 4,
    'Tipo 5': 5
})
#%%
# Treinar o modelo
model.fit(X, y)

#%%

plt.figure(dpi=400)
tree.plot_tree(model,
               feature_names=features,
               class_names=model.classes_,
               filled=True, max_depth=3)
