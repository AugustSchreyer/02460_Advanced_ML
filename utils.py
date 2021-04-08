import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_sequences(model, PLOT_DATA, NUM_PLOTS=9, ANOMALY_THRESHOLD=0.1, samples=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_val = PLOT_DATA.max()
    min_val = PLOT_DATA.min()
    if samples is None:
        samples = np.random.choice(len(PLOT_DATA), replace=False, size=NUM_PLOTS)
    fig, axes = plt.subplots(nrows=int(np.sqrt(NUM_PLOTS)), ncols=int(np.sqrt(NUM_PLOTS)))
    fig.tight_layout()
    fig.set_size_inches(16, 9)
    for i, ax in enumerate(axes.flat):

        dataset_idx = samples[i]
        
        dat = PLOT_DATA[dataset_idx].to(device).unsqueeze(0)
        outputs = model.forward(dat)

        log_probs = outputs["px"].log_prob(dat.view(-1)).exp().detach().cpu().numpy()
        idx = log_probs < ANOMALY_THRESHOLD
        anom = np.arange(len(PLOT_DATA[dataset_idx]))[idx.squeeze()]
        
        mu = outputs["px"].mu.view(-1).detach().cpu().numpy()
        sigma = outputs["px"].sigma.view(-1).detach().cpu().numpy()
        ax.plot(mu, c="r", linewidth=2)
        ax.fill_between(list(range(len(mu))), mu-2*sigma, mu+2*sigma, facecolor='red', alpha=0.5)
        ax.plot(dat.view(-1).detach().cpu().numpy(), c="b", linewidth=2)
        ax.scatter(anom, dat.view(-1).detach().cpu().numpy()[anom], c="r", s=50, label="anomaly")
        
        ax.set_title(f"Sample {samples[i]}", fontdict={"size": 12})
        ax.set_ylim(min_val-2*sigma.max(), max_val+2*sigma.max())
        plt.setp(ax.get_xticklabels(), fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)
    plt.show()
    