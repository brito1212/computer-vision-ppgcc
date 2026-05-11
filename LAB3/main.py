"""
Laboratório 03 - Granulometria Morfológica
PPGCC - Visão Computacional. Prof. Prof. Dr. Fábio Cappabianco
Aluno: Felipe Faustino Brito

Calcula assinaturas granulométricas de imagens em tons de cinza usando
operações morfológicas (erosão/dilatação) em múltiplas escalas, compara
as classes e gera relatório visual completo.

Estrutura esperada do diretório de imagens:
    images/
        classe1/  (10-20 imagens .jpg/.png/.bmp)
        classe2/
        classe3/

Uso:
    python main.py
    python main.py --images_dir meu_folder --output_dir resultados
"""

import os
import sys
import argparse
import warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import ndimage
from skimage import io, color, transform, morphology
from skimage.morphology import disk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PARÂMETROS GLOBAIS
# ──────────────────────────────────────────────────────────────────────────────
IMG_SIZE = (128, 128)  # tamanho de redimensionamento
SCALES = list(range(1, 21))  # raios dos elementos estruturantes (1..20)
SE_SHAPE = "disk"  # "disk" ou "square"
GRANULO_OP = "opening"  # "opening" (padrão) ou "closing"
N_COLS_GRID = 5  # colunas na grade de exemplos

COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"]


# ──────────────────────────────────────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ──────────────────────────────────────────────────────────────────────────────


def load_image_gray(path: Path, size=IMG_SIZE) -> np.ndarray | None:
    """Carrega imagem, converte para cinza e redimensiona."""
    try:
        img = io.imread(str(path))
        if img.ndim == 3:
            img = color.rgb2gray(img)  # [0,1]
        elif img.dtype == np.uint8:
            img = img.astype(np.float64) / 255.0
        else:
            img = img.astype(np.float64)
            if img.max() > 1.0:
                img /= img.max()
        img = transform.resize(img, size, anti_aliasing=True)
        return img.astype(np.float32)
    except Exception as e:
        print(f"  [AVISO] Não foi possível carregar {path.name}: {e}")
        return None


def make_se(radius: int, shape: str = SE_SHAPE) -> np.ndarray:
    """Cria elemento estruturante."""
    if shape == "disk":
        return disk(radius).astype(bool)
    else:  # square
        s = 2 * radius + 1
        return np.ones((s, s), dtype=bool)


def granulometric_signature(
    img: np.ndarray, scales=SCALES, operation=GRANULO_OP
) -> np.ndarray:
    original_mass = float(img.sum())
    if original_mass == 0:
        return np.zeros(len(scales), dtype=np.float32)

    sig = np.zeros(len(scales), dtype=np.float32)
    prev_mass = original_mass

    for i, r in enumerate(scales):
        se = make_se(r)

        if operation == "opening":
            filtered = morphology.opening(img, se)
            curr_mass = float(filtered.sum())
            sig[i] = (prev_mass - curr_mass) / original_mass
            prev_mass = curr_mass

        elif operation == "closing":
            filtered = morphology.closing(img, se)
            curr_mass = float(filtered.sum())
            sig[i] = (curr_mass - prev_mass) / original_mass  # closing aumenta a massa
            prev_mass = curr_mass

        elif operation == "gradient":
            dilated = morphology.dilation(img, se)
            eroded = morphology.erosion(img, se)
            gradient = dilated - eroded
            curr_mass = float(gradient.sum())
            sig[i] = curr_mass / original_mass  # massa do gradiente normalizada
            # sem diferença acumulada: cada escala é independente

    return sig


def load_class(class_dir: Path, verbose=True):
    """Carrega todas as imagens de uma classe."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pgm"}
    files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in exts])
    images, names = [], []
    for f in files:
        img = load_image_gray(f)
        if img is not None:
            images.append(img)
            names.append(f.name)
    if verbose:
        print(f"  Classe '{class_dir.name}': {len(images)} imagens carregadas")
    return images, names


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZAÇÕES
# ──────────────────────────────────────────────────────────────────────────────


def plot_example_images(classes_data: dict, output_dir: Path, n_cols=N_COLS_GRID):
    """Grade de exemplos de imagens por classe."""
    class_names = list(classes_data.keys())
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_cols, figsize=(n_cols * 2.2, n_classes * 2.4))
    fig.suptitle(
        "Exemplos de Imagens por Classe", fontsize=14, fontweight="bold", y=1.01
    )

    for ci, cname in enumerate(class_names):
        imgs = classes_data[cname]["images"][:n_cols]
        for j in range(n_cols):
            ax = axes[ci, j] if n_classes > 1 else axes[j]
            if j < len(imgs):
                ax.imshow(imgs[j], cmap="gray", vmin=0, vmax=1)
            else:
                ax.axis("off")
                continue
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(
                    cname, fontsize=10, fontweight="bold", rotation=90, labelpad=6
                )
    plt.tight_layout()
    out = output_dir / "01_exemplos_imagens.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out.name}")


def plot_individual_signatures(classes_data: dict, output_dir: Path):
    """Assinaturas individuais de cada imagem, coloridas por classe."""
    n_classes = len(classes_data)
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 4.5), sharey=True)
    if n_classes == 1:
        axes = [axes]

    for ci, (cname, data) in enumerate(classes_data.items()):
        ax = axes[ci]
        color = COLORS[ci % len(COLORS)]
        sigs = data["signatures"]
        for sig in sigs:
            ax.plot(SCALES, sig, color=color, alpha=0.4, linewidth=0.9)
        mean_sig = np.mean(sigs, axis=0)
        ax.plot(SCALES, mean_sig, color="black", linewidth=2.2, label="Média", zorder=5)
        ax.set_title(cname, fontsize=12, fontweight="bold")
        ax.set_xlabel("Escala (raio SE)", fontsize=10)
        ax.set_ylabel("Diferença Granulométrica Normalizada", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(SCALES[0], SCALES[-1])

    fig.suptitle(
        "Assinaturas Granulométricas Individuais por Classe",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out = output_dir / "02_assinaturas_individuais.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out.name}")


def plot_mean_signatures(classes_data: dict, output_dir: Path):
    """Assinaturas médias de todas as classes sobrepostas + desvio padrão."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Gráfico de linha com banda de std
    for ci, (cname, data) in enumerate(classes_data.items()):
        color = COLORS[ci % len(COLORS)]
        sigs = np.array(data["signatures"])
        mean = sigs.mean(axis=0)
        std = sigs.std(axis=0)
        ax1.plot(SCALES, mean, color=color, linewidth=2.2, label=cname)
        ax1.fill_between(SCALES, mean - std, mean + std, color=color, alpha=0.15)

    ax1.set_title("Assinatura Média por Classe (± 1σ)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Escala (raio SE)", fontsize=10)
    ax1.set_ylabel("Diferença Granulométrica Normalizada", fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(SCALES[0], SCALES[-1])

    # ── Assinatura cumulativa (granulometria acumulada)
    for ci, (cname, data) in enumerate(classes_data.items()):
        color = COLORS[ci % len(COLORS)]
        sigs = np.array(data["signatures"])
        cumul = np.cumsum(sigs, axis=1)
        mean_c = cumul.mean(axis=0)
        ax2.plot(SCALES, mean_c, color=color, linewidth=2.2, label=cname)

    ax2.set_title(
        "Assinatura Granulométrica Cumulativa Média", fontsize=12, fontweight="bold"
    )
    ax2.set_xlabel("Escala (raio SE)", fontsize=10)
    ax2.set_ylabel("Soma Acumulada", fontsize=9)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(SCALES[0], SCALES[-1])

    plt.tight_layout()
    out = output_dir / "03_assinaturas_medias.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out.name}")


def plot_pca(classes_data: dict, output_dir: Path):
    """Redução PCA 2D das assinaturas para visualizar separabilidade."""
    all_sigs, all_labels = [], []
    for cname, data in classes_data.items():
        for sig in data["signatures"]:
            all_sigs.append(sig)
            all_labels.append(cname)

    X = np.array(all_sigs)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sc)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for ci, cname in enumerate(classes_data.keys()):
        idx = [i for i, l in enumerate(all_labels) if l == cname]
        ax.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            color=COLORS[ci % len(COLORS)],
            label=cname,
            s=70,
            edgecolors="white",
            linewidths=0.5,
        )

    var = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=10)
    ax.set_title("PCA das Assinaturas Granulométricas", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = output_dir / "04_pca.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out.name}")
    return X_pca, all_labels, var


def plot_distance_matrix(classes_data: dict, output_dir: Path):
    """Matriz de distâncias euclidianas entre assinaturas médias."""
    cnames = list(classes_data.keys())
    means = [np.mean(classes_data[c]["signatures"], axis=0) for c in cnames]
    n = len(cnames)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(means[i] - means[j])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(D, cmap="YlOrRd_r", vmin=0)
    plt.colorbar(im, ax=ax, label="Distância Euclidiana")
    ax.set_xticks(range(n))
    ax.set_xticklabels(cnames, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(cnames)
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{D[i,j]:.4f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black" if D[i, j] > D.max() * 0.4 else "white",
            )
    ax.set_title(
        "Distância Euclidiana entre Médias das Classes", fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out = output_dir / "05_matriz_distancias.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out.name}")
    return D, cnames


def plot_morphological_sequence(classes_data: dict, output_dir: Path):
    """
    Para a primeira imagem de cada classe, mostra a imagem original +
    aberturas em escalas selecionadas.
    """
    sample_scales = [1, 3, 7, 12, 20]
    n_cls = len(classes_data)
    n_cols = 1 + len(sample_scales)

    fig, axes = plt.subplots(n_cls, n_cols, figsize=(n_cols * 2.2, n_cls * 2.4))
    if n_cls == 1:
        axes = [axes]

    col_titles = ["Original"] + [f"r={r}" for r in sample_scales]
    for j, t in enumerate(col_titles):
        axes[0][j].set_title(t, fontsize=10, fontweight="bold")

    for ci, (cname, data) in enumerate(classes_data.items()):
        img = data["images"][0]
        row_imgs = [img]
        for r in sample_scales:
            se = make_se(r)
            row_imgs.append(morphology.opening(img, se))

        for j, im in enumerate(row_imgs):
            axes[ci][j].imshow(im, cmap="gray", vmin=0, vmax=1)
            axes[ci][j].set_xticks([])
            axes[ci][j].set_yticks([])
            if j == 0:
                axes[ci][j].set_ylabel(cname, fontsize=10, fontweight="bold")

    fig.suptitle(
        "Efeito da Abertura Morfológica em Diferentes Escalas",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    out = output_dir / "06_sequencia_morfologica.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out.name}")


def plot_boxplot(classes_data: dict, output_dir: Path):
    """Box-plots das assinaturas em escalas selecionadas."""
    sample_scales_idx = [0, 2, 4, 9, 14, 19]  # índices de SCALES
    sample_labels = [f"r={SCALES[i]}" for i in sample_scales_idx]

    n_scales = len(sample_scales_idx)
    fig, axes = plt.subplots(1, n_scales, figsize=(3.5 * n_scales, 4.5), sharey=False)

    for si, (sidx, slabel) in enumerate(zip(sample_scales_idx, sample_labels)):
        ax = axes[si]
        data_per_class = []
        cnames = []
        for ci, (cname, data) in enumerate(classes_data.items()):
            vals = [sig[sidx] for sig in data["signatures"]]
            data_per_class.append(vals)
            cnames.append(cname)

        bp = ax.boxplot(
            data_per_class,
            patch_artist=True,
            notch=False,
            medianprops=dict(color="black", linewidth=1.8),
        )
        for patch, color in zip(bp["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(cnames) + 1))
        ax.set_xticklabels(cnames, rotation=30, ha="right", fontsize=9)
        ax.set_title(slabel, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if si == 0:
            ax.set_ylabel("Diferença Granulométrica", fontsize=9)

    fig.suptitle(
        "Distribuição das Assinaturas por Escala Selecionada",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    out = output_dir / "07_boxplots.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# RELATÓRIO TEXTUAL
# ──────────────────────────────────────────────────────────────────────────────


def print_stats(classes_data: dict, dist_matrix, cnames):
    print("\n" + "═" * 60)
    print("  ESTATÍSTICAS DAS ASSINATURAS GRANULOMÉTRICAS")
    print("═" * 60)
    for cname, data in classes_data.items():
        sigs = np.array(data["signatures"])
        print(f"\n  Classe: {cname}")
        print(f"    Imagens     : {len(data['images'])}")
        print(f"    Média (µ)   : {sigs.mean():.5f}")
        print(f"    Desvio (σ)  : {sigs.std():.5f}")
        print(f"    Min / Max   : {sigs.min():.5f} / {sigs.max():.5f}")
        # escala de maior resposta
        peak_scale = SCALES[sigs.mean(axis=0).argmax()]
        print(f"    Pico médio  : escala r={peak_scale}")

    print("\n  Distâncias Euclidianas entre Médias:")
    n = len(cnames)
    for i in range(n):
        for j in range(i + 1, n):
            print(f"    {cnames[i]} ↔ {cnames[j]}: {dist_matrix[i, j]:.5f}")

    if n >= 2:
        flat = []
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                flat.append(dist_matrix[i, j])
                pairs.append((cnames[i], cnames[j]))
        best_pair = pairs[int(np.argmax(flat))]
        worst_pair = pairs[int(np.argmin(flat))]
        print(f"\n  Par mais separável : {best_pair[0]} ↔ {best_pair[1]}")
        print(f"  Par mais similar   : {worst_pair[0]} ↔ {worst_pair[1]}")
    print("═" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Granulometria Morfológica - Lab 03")
    parser.add_argument(
        "--images_dir", default="images", help="Diretório raiz com subpastas de classes"
    )
    parser.add_argument(
        "--output_dir",
        default="resultados_granulometria",
        help="Diretório de saída para figuras",
    )
    parser.add_argument(
        "--operation",
        default=GRANULO_OP,
        choices=["opening", "closing", "gradient"],
        help="Operação morfológica base",
    )
    parser.add_argument(
        "--se_shape",
        default=SE_SHAPE,
        choices=["disk", "square"],
        help="Forma do elemento estruturante",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  Laboratório 03 – Granulometria Morfológica")
    print(f"  Operação: {args.operation} | SE: {args.se_shape}")
    print(f"  Escalas : r = {SCALES[0]} … {SCALES[-1]}")
    print(f"  Imagens : {images_dir}")
    print(f"  Saída   : {output_dir}")
    print(f"{'═'*60}\n")

    # ── 1. Descobrir classes
    class_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    if len(class_dirs) < 2:
        sys.exit(
            f"[ERRO] Encontradas {len(class_dirs)} subpastas em '{images_dir}'. "
            f"São necessárias pelo menos 2."
        )
    print(f"Classes encontradas ({len(class_dirs)}): {[d.name for d in class_dirs]}\n")

    # ── 2. Carregar imagens e calcular assinaturas
    classes_data = {}
    for cd in class_dirs:
        print(f"Processando classe: {cd.name}")
        images, names = load_class(cd)
        if len(images) == 0:
            print(f"  [AVISO] Nenhuma imagem válida em {cd.name}, pulando.")
            continue

        sigs = []
        for k, img in enumerate(images):
            sig = granulometric_signature(img, scales=SCALES, operation=args.operation)
            sigs.append(sig)
            if (k + 1) % 5 == 0 or (k + 1) == len(images):
                print(f"  [{k+1}/{len(images)}] assinaturas calculadas...", end="\r")
        print()

        classes_data[cd.name] = {
            "images": images,
            "names": names,
            "signatures": sigs,
        }

    if len(classes_data) < 2:
        sys.exit(
            "[ERRO] Menos de 2 classes com imagens válidas. Verifique o diretório."
        )

    # ── 3. Gerar figuras
    print("\nGerando figuras...")
    plot_example_images(classes_data, output_dir)
    plot_individual_signatures(classes_data, output_dir)
    plot_mean_signatures(classes_data, output_dir)
    plot_pca(classes_data, output_dir)
    dist_matrix, cnames = plot_distance_matrix(classes_data, output_dir)
    plot_morphological_sequence(classes_data, output_dir)
    plot_boxplot(classes_data, output_dir)

    # ── 4. Estatísticas
    print_stats(classes_data, dist_matrix, cnames)

    print(f"Concluído! Todos os resultados estão em: {output_dir}/\n")


if __name__ == "__main__":
    main()
