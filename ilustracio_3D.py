import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn import svm
from sklearn.datasets import make_blobs 

# --------------------------------------------------------------------------
# Pas 1: Generem dades
# --------------------------------------------------------------------------
def generar_dades_3d(n_samples=150, n_features=3, centers=2):
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features,
                      random_state=42, cluster_std=1.1) 
    y[y == 0] = -1
    return X, y

num_samples_plot = 200
X_train_3d, y_train_3d = generar_dades_3d(n_samples=num_samples_plot, n_features=3)

print(f"Dimensions de X_train_3d: {X_train_3d.shape}")
print(f"Dimensions de y_train_3d: {y_train_3d.shape}")
print(f"Classes úniques a y_train_3d: {np.unique(y_train_3d)}")

# --------------------------------------------------------------------------
# Pas 2: Entrenar el model SVM
# --------------------------------------------------------------------------

nu_param = 0.1  # 10% d'errors permesos

model_3d = svm.NuSVC(nu=nu_param, kernel='linear') 
model_3d.fit(X_train_3d, y_train_3d)

# --------------------------------------------------------------------------
# Pas 3: Extreure informació
# --------------------------------------------------------------------------
w = model_3d.coef_[0]
b = model_3d.intercept_[0]
support_vectors_3d = model_3d.support_vectors_

print(f"\nPesos (w): {w}")
print(f"Intercept (b): {b}")
print(f"Nº de Vectors de suport: {len(support_vectors_3d)}")
if X_train_3d.shape[0] > 0 and len(support_vectors_3d) > 0 :
     print(f"Fracció de vectors de suport: {len(support_vectors_3d) / X_train_3d.shape[0]:.3f}")


# --------------------------------------------------------------------------
# Pas 4: Generem la visualització
# --------------------------------------------------------------------------
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter_plot = ax.scatter(X_train_3d[:, 0], X_train_3d[:, 1], X_train_3d[:, 2],
                          c=y_train_3d, s=50, cmap='winter', alpha=0.7, edgecolors='k')

ax.set_xlabel("Característica 1 (X1)")
ax.set_ylabel("Característica 2 (X2)")
ax.set_zlabel("Característica 3 (X3)")
plt.title(f"Visualització d'una SVC lineal en 3D (nu={model_3d.get_params()['nu']:.2f})") 

if len(support_vectors_3d) > 0:
    ax.scatter(support_vectors_3d[:, 0], support_vectors_3d[:, 1], support_vectors_3d[:, 2],
               s=150, linewidth=1.5, facecolors='none', edgecolors='red', label='Vectors de suport')

if abs(w[2]) > 1e-6:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if not np.all(np.isfinite(xlim)) or not np.all(np.isfinite(ylim)):
        print("Límits d'eixos invàlids")
        xlim = (-3,3) if not np.all(np.isfinite(xlim)) else xlim
        ylim = (-3,3) if not np.all(np.isfinite(ylim)) else ylim

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 10),
                         np.linspace(ylim[0], ylim[1], 10))

    zz_plane = (-w[0] * xx - w[1] * yy - b) / w[2]
    ax.plot_surface(xx, yy, zz_plane, alpha=0.4, color='purple', rstride=100, cstride=100)

    zz_margin_plus = (-w[0] * xx - w[1] * yy - b + 1) / w[2] # Marge per clase +1
    ax.plot_surface(xx, yy, zz_margin_plus, alpha=0.2, color='blue', rstride=100, cstride=100)

    zz_margin_minus = (-w[0] * xx - w[1] * yy - b - 1) / w[2] # Marge per clase -1
    ax.plot_surface(xx, yy, zz_margin_minus, alpha=0.2, color='orange', rstride=100, cstride=100)
else:
    print("Surt un pla vertical")


# Leyendas
handles, labels = scatter_plot.legend_elements()
if len(np.unique(y_train_3d)) == 2 and len(handles) > 0 :
    class_labels = [f"Classe {int(l)}" for l in np.unique(y_train_3d)]
    # No hi ha més etiquetes que handles
    num_unique_classes = len(class_labels)
    if len(handles) >= num_unique_classes:
        legend1 = ax.legend(handles[:num_unique_classes], class_labels, title="Classes", loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.add_artist(legend1) 

if len(support_vectors_3d) > 0:
    from matplotlib.lines import Line2D
    sv_legend_handle = Line2D([0], [0], marker='o', color='w', label='Vectors de suport',
                              markerfacecolor='none', markeredgecolor='red', markersize=10)
    ax.legend(handles=[sv_legend_handle], loc="upper left", bbox_to_anchor=(1.02, 0.8))


plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar layout 
plt.show()