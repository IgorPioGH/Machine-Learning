#%%
import pandas as pd
from sklearn import tree

# %%
df = pd.read_excel("data/dados_cerveja.xlsx")
df
# %%
features = ['temperatura', 'copo', 'espuma', 'cor']
target = 'classe'

y = df[target]

X = df[features]

#%%
# Para o SikcitLearn todas as variáveis devem ser valores numéricos
# Passando as variáveis para o tipo numérico (Variáveis Dummies)
X = X.replace({
    'mud':1,
    'pint':2,
    'não':0,
    'sim':1,
    'escura':1,
    'clara':0
})
#%%
model = tree.DecisionTreeClassifier(random_state=42)

model.fit(X=X, y=y)

#%%
tree.plot_tree(model,
               feature_names=features,
               class_names=model.classes_,
               filled=True)
