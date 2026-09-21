"""Render the deck's LaTeX equations to transparent PNGs via matplotlib mathtext."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "eq")
os.makedirs(OUT, exist_ok=True)

INK = "#21295C"      # midnight navy — reads on light cards
ACCENT = "#B24C2E"   # terracotta for the "on one fixed graph" emphasis

EQUATIONS = {
    "gdar":        (r"$x(t{+}1)=\sum_{k=0}^{K-1} A_k\, x(t{-}k) + B\, u(t) + e(t)$", INK, 30),
    "support":     (r"$\mathrm{supp}(A_k)\subseteq E\cup\mathrm{diag}\qquad A_k = a_k I - b_k L$", INK, 28),
    "tp_param":    (r"$A_k(p)=\sum_{j} w_j(p)\, A_k^{(j)}\,,\quad w_j(p)\geq 0,\ \sum_j w_j(p)=1$", INK, 28),
    "hosvd":       (r"$\mathcal{A}(p)\;\approx\;\mathcal{S}\;\times_1 U_1(p)\;\cdots\;\times_d U_d(p)$", INK, 28),
    "polytope":    (r"$A(p)=\sum_{v=1}^{V}\lambda_v(p)\, A_v\,,\quad \lambda_v\geq 0,\ \sum_v\lambda_v=1$", INK, 28),
    "eig":         (r"$A_k(p)=a_k(p)\,I-b_k(p)\,L\;\Rightarrow\;$ shared eigenvectors of $L$, moving $\lambda_i(p)$", INK, 24),
    "product":     (r"$\mathcal{G}_{\mathrm{space}}\;\otimes\;\mathcal{G}_{\mathrm{pop}}$", ACCENT, 40),
    "latent":      (r"$x(t{+}1)=A\,x(t)+w,\qquad y(t)=C\,x(t)+v$", INK, 30),
    "lmi":         (r"$\mathrm{find}\ Q\succ 0,\,Y:\ \ (A_v Q + B Y)\ \mathrm{Schur}\ \ \forall v\,,\quad F=YQ^{-1}$", INK, 24),
    "dale":        (r"$A_{ab}\geq 0$ if source $b$ excitatory$,\quad A_{ab}\leq 0$ if inhibitory", INK, 22),
    "recover":     (r"$\hat{A}(p)=\sum_{v=1}^{5}\lambda_v(p)\,\hat{A}_v\;\;\longleftrightarrow\;\;$ true 5 graphs?", ACCENT, 26),
}


def render(name, tex, color, fs):
    fig = plt.figure(figsize=(0.1, 0.1))
    t = fig.text(0.0, 0.0, tex, fontsize=fs, color=color, ha="left", va="bottom")
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=220, transparent=True,
                bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


if __name__ == "__main__":
    for name, (tex, color, fs) in EQUATIONS.items():
        render(name, tex, color, fs)
        print("rendered", name)
    print("->", OUT)
