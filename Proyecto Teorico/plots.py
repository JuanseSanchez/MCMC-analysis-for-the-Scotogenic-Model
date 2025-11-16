import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('/home/js.sanchezl1/Proyecto_Teorico/only_unitarity_perturbativity.csv', sep=';')


m_etaI = data["MASS etI"]
m_N1    = data["MNIN Mn(1,1)"]
relic_density = data["MICROMEGAS Omega h^2 (Dark matter relic density)"]
lambda1 = np.array(data["MINPAR lambda1Input"])
lambda2 = np.array(data["MINPAR lambda2Input"])
lambda3 = np.array(data["MINPAR lambda3Input"])
lambda4 = np.array(data["MINPAR lambda4Input"])
lambda5 = np.array(data["MINPAR lambda5Input"])

# Extraer los acoplamientos de Yukawa (parte real e imaginaria)
yukawa_labels = [
    ("YNIN Yn(1,1)", "IMYNIN Yn(1,1)"),
    ("YNIN Yn(1,2)", "IMYNIN Yn(1,2)"),
    ("YNIN Yn(1,3)", "IMYNIN Yn(1,3)"),
    ("YNIN Yn(2,1)", "IMYNIN Yn(2,1)"),
    ("YNIN Yn(2,2)", "IMYNIN Yn(2,2)"),
    ("YNIN Yn(2,3)", "IMYNIN Yn(2,3)"),
    ("YNIN Yn(3,1)", "IMYNIN Yn(3,1)"),
    ("YNIN Yn(3,2)", "IMYNIN Yn(3,2)"),
    ("YNIN Yn(3,3)", "IMYNIN Yn(3,3)")
]

# Calcular la norma de cada elemento de Yukawa
yukawa_norms = []
for re_label, im_label in yukawa_labels:
    re_part = np.array(data[re_label])
    im_part = np.array(data[im_label])
    norm = np.sqrt(re_part**2 + im_part**2)
    yukawa_norms.append(norm)

# Convertir a array numpy (shape: 9 x N_points)
yukawa_norms = np.array(yukawa_norms)

# Calcular la media geométrica de la norma de los Yukawa para cada punto
# Media geométrica = (producto de todos los elementos)^(1/n)
geometric_mean_yukawa = np.prod(yukawa_norms, axis=0)**(1.0/len(yukawa_labels))

# Filtra valores finitos (evita NaN/inf)
mask = np.isfinite(relic_density) & np.isfinite(m_etaI) & np.isfinite(m_N1) & np.isfinite(geometric_mean_yukawa) & (geometric_mean_yukawa > 0)

plt.figure()
plt.scatter(m_etaI[mask], relic_density[mask], s=20, alpha=0.7, marker='o', label="Masa η_I")
plt.scatter(m_N1[mask],   relic_density[mask], s=20, alpha=0.7, marker='x', label="Masa N1")
plt.xlabel("Masa [GeV]")
plt.ylabel("Densidad de reliquia")
plt.title("Densidad de reliquia vs masas: η_I y N1")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale("log")
plt.yscale("log")
# Guarda en la misma carpeta de tu script
plt.savefig("scatter_omega_vs_etaI_y_N1.png", dpi=300, bbox_inches="tight")
plt.close()
# Histograma de λ1–λ4
plt.hist(lambda2, bins=100, alpha=0.6, label="λ2", color="blue", density=True)
plt.hist(lambda3, bins=100, alpha=0.6, label="λ3", color="orange", density=True)
plt.hist(lambda4, bins=100, alpha=0.6, label="λ4", color="green", density=True)
plt.xlabel("Valor de λ")
plt.ylabel("Frecuencia")
plt.title("Distribución de λ2–λ4")
plt.legend()
#plt.xlim(0,0.8)
plt.xscale('log')
plt.savefig("hist_lambdas_2_to_4.png", dpi=300, bbox_inches="tight")
plt.close()

# Histograma de λ5
plt.hist(
    lambda5,
    bins=5000,
    alpha=0.6,
    label="λ5",
    color="black",
    density=True,   
)
plt.xlabel("Valor de λ5")
plt.ylabel("Frecuencia")
plt.title("Distribución de λ5")
plt.legend()
plt.xscale('log')
plt.savefig("hist_lambda5.png", dpi=300, bbox_inches="tight")
plt.close()

# Histograma de la media geométrica de la norma de los Yukawa
plt.figure()
plt.hist(
    geometric_mean_yukawa[mask],
    bins=100,
    alpha=0.7,
    label="Media geométrica de ||Yukawa||",
    color="purple",
    density=True,
)
plt.xlabel("Media geométrica de ||Yᵢⱼ||")
plt.ylabel("Frecuencia normalizada")
plt.title("Distribución de la media geométrica de la norma de los Yukawa")
plt.legend()
#plt.xscale('log')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig("hist_geometric_mean_yukawa.png", dpi=300, bbox_inches="tight")
plt.close()

# Scatter plots de las normas de los Yukawa
# yukawa_norms tiene shape (9, N_points)
# Índices: Y11=0, Y12=1, Y13=2, Y21=3, Y22=4, Y23=5, Y31=6, Y32=7, Y33=8

# Crear carpeta para los scatter plots
import os
scatter_dir = "scatter_plots_yukawa"
os.makedirs(scatter_dir, exist_ok=True)

yukawa_names = [
    "Y₁₁", "Y₁₂", "Y₁₃",
    "Y₂₁", "Y₂₂", "Y₂₃",
    "Y₃₁", "Y₃₂", "Y₃₃"
]

# Crear scatter plots comparando solo Y1j vs Yij (excepto primera fila)
# Y11 (idx=0), Y12 (idx=1), Y13 (idx=2) vs Y2j y Y3j solamente
comparisons = []
for idx_first_row in [0, 1, 2]:  # Y11, Y12, Y13
    for idx_other in range(3, 9):   # Solo Y21, Y22, Y23, Y31, Y32, Y33
        comparisons.append((idx_first_row, idx_other))


for idx_i, idx_j in comparisons:
    plt.figure(figsize=(8, 6))
    
    # Crear máscara para valores finitos y positivos
    mask_ij = (yukawa_norms[idx_i] > 0) & (yukawa_norms[idx_j] > 0) & \
              np.isfinite(yukawa_norms[idx_i]) & np.isfinite(yukawa_norms[idx_j])
    
    plt.scatter(
        yukawa_norms[idx_i][mask_ij],
        yukawa_norms[idx_j][mask_ij],
        s=10,
        alpha=0.5,
        c='blue'
    )
    
    plt.xlabel(f"|{yukawa_names[idx_i]}|")
    plt.ylabel(f"|{yukawa_names[idx_j]}|")
    plt.title(f"Scatter plot: |{yukawa_names[idx_i]}| vs |{yukawa_names[idx_j]}|")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Calcular correctamente los índices (i,j) de la matriz 3x3
    row_i = (idx_i // 3) + 1  # fila: 1, 2, o 3
    col_i = (idx_i % 3) + 1   # columna: 1, 2, o 3
    row_j = (idx_j // 3) + 1
    col_j = (idx_j % 3) + 1
    
    filename = f"scatter_yukawa_Y{row_i}{col_i}_vs_Y{row_j}{col_j}.png"
    filepath = os.path.join(scatter_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


