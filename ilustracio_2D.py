import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs 

# --------------------------------------------------------------------------
# Pas 1: Generar dades
# --------------------------------------------------------------------------
def generar_datos_simulados(n_samples=100, n_features=2, centers=2):
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features,
                      random_state=42, cluster_std=1.2) 
    y[y == 0] = -1 # Labels
    return X, y

# Generem dades 2D
num_samples_plot = 200
X_train, y_train = generar_datos_simulados(n_samples=num_samples_plot, n_features=2)

print(f"Dimensions de X_train: {X_train.shape}")
print(f"Dimensions de y_train: {y_train.shape}")
print(f"Classes úniques a y_train: {np.unique(y_train)}")

# --------------------------------------------------------------------------
# Pas 2: Entrenem el model
# --------------------------------------------------------------------------

nu_param = 0.1 

model = svm.NuSVC(nu=nu_param, kernel='linear')
model.fit(X_train, y_train)

# --------------------------------------------------------------------------
# Pas 3: Vectors de suport
# --------------------------------------------------------------------------
# Coeficients del pla (pesos w)
w = model.coef_[0]

# Intercept (b)
b = model.intercept_[0]

# Support vectors
support_vectors = model.support_vectors_
support_vector_indices = model.support_ # Índices en X_train

print(f"\nPesos (w): {w}")
print(f"Intercept (b): {b}")
print(f"Número de vectors de suport: {len(support_vectors)}")
if len(support_vectors) > 0:
    print(f"Fracció de vectors de suport: {len(support_vectors) / X_train.shape[0]:.2f}")
else:
    print("No hi ha vectors de suport")


# --------------------------------------------------------------------------
# Pas 4: Generem la visualització
# --------------------------------------------------------------------------
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='winter', alpha=0.7, edgecolors='k')
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
if isinstance(model, svm.NuSVC):
    plt.title(f"Visualització d'una SVC lineal en 2D (nu={model.get_params()['nu']:.2f})")
elif isinstance(model, svm.SVC):
    plt.title(f"Visualització d'una SVC lineal en 2D (C={model.get_params()['C']})")


if len(support_vectors) > 0:
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=150,
                linewidth=1.5, facecolors='none', edgecolors='red', label='Vectors de suport')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 50)
yy = np.linspace(ylim[0], ylim[1], 50)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

contour = ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.8,
                     linestyles=['--', '-', '--'])
ax.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

handles, labels = scatter.legend_elements()
if len(np.unique(y_train)) == 2 and len(handles) > 0 :
    class_labels = [f"Classe {int(l)}" for l in np.unique(y_train)]
    if len(handles) == len(class_labels):
         legend1 = ax.legend(handles, class_labels, title="Classes", loc="upper right")
         ax.add_artist(legend1)

plt.legend(loc='lower left') # Llegenda
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()