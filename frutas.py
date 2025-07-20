#%%
import pandas as pd
from sklearn import tree
#%%
df = pd.read_excel("data/dados_frutas.xlsx")
df

#%%

arvore = tree.DecisionTreeClassifier(random_state=42)

# %%
# resposta (Serie)
y = df['Fruta']

caracteristicas = ['Arredondada', 'Suculenta', 'Vermelha', 'Doce']
X = df[caracteristicas]
print(X)

#%%
# Ajustar o modelo -> Ensinar a máquina
# Passando as covariáveis e as respostas
# Essa parte é o Machine Learning!
arvore.fit(X, y)

#%%
# Cada elemento corresponde a um atributo
arvore.predict([[0,0,0,0]])

#%% 
import matplotlib.pyplot as plt
plt.figure(dpi=400)

# imprimindo a arvore

tree.plot_tree(arvore, 
               feature_names=caracteristicas,
               class_names=arvore.classes_,
               filled=True)


#%%
# retorna a probabilidade de cada uma das classes
proba = arvore.predict_proba([[1,1,1,1]])[0]
pd.Series(proba, index=arvore.classes_)
