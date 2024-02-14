import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee"])

KGW_KGW = [52.80, 19.80]
KGW_SIR = [3.40, 90.09]
SIR_PRW = [41.05, 19.80]
PRW_KGW = [37.00, 28.20]

data = [KGW_KGW, KGW_SIR, SIR_PRW, PRW_KGW] * 2
labels = ["KGW_KGW", "KGW_SIR", "SIR_PRW", "PRW_KGW"] * 2
fig, ax = plt.subplots()
pos = []
for d in zip(range(len(labels)), data, labels):
    # fig, ax = plt.subplots()
    offset = 0 if d[0] < 4 else 0.5
    ax.set_ylim(-105, 105)
    ax.bar(x=d[0] + offset, height=d[1][0], align="center", color="#33BBEE")
    ax.text(d[0] + offset, d[1][0], "%.2f" % d[1][0], ha="center", va="bottom", size=6)
    ax.bar(x=d[0] + offset, height=-d[1][1], align="center", color="#FFDD77")
    ax.text(d[0] + offset, -d[1][1] - 10, "%.2f" % d[1][1], ha="center", va="bottom", size=6)
    if d[0] == 3:
        tmp_x = [d[0] + 0.75, d[0] + 0.75]
        tmp_y = [-1000, 1000]
        ax.plot(tmp_x, tmp_y, linewidth=1, linestyle="dashed")
    pos.append(d[0] + offset)

blue_patch = mpatches.Patch(color="#33BBEE", label="$D_W$")
yellow_patch = mpatches.Patch(color="#FFDD77", label="$D_P$")
ax.legend(handles=[blue_patch, yellow_patch], loc="lower right", fontsize=5)

ax.tick_params(axis="both", which="both", length=0)
ax.set_xticks(pos, labels, fontsize=6, rotation=-65)
ax.text(1 + 0.5, 80, "LLama-2-13B", ha="center", va="bottom")
ax.text(5 + 1.0, 80, "OPT-1.3B", ha="center", va="bottom")

ax.get_yaxis().set_visible(False)
# ax.get_xaxis().set_visible(False)
fig.savefig("KGW_KGW.png")
