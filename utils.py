import torch
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy

def voigt(x,mu,sigma,gamma,z):
    y = scipy.special.voigt_profile(x-mu,sigma,gamma)
    y = z*y/np.max(y)
    return y

def plot_sequences(model, PLOT_DATA, NUM_PLOTS=9, ANOMALY_THRESHOLD=0.1, samples=None):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #max_val = PLOT_DATA.max()
    #min_val = PLOT_DATA.min()
    if samples is None:
        samples = np.random.choice(len(PLOT_DATA), replace=False, size=NUM_PLOTS)
    fig, axes = plt.subplots(nrows=int(np.sqrt(NUM_PLOTS)), ncols=int(np.sqrt(NUM_PLOTS)))
    fig.set_size_inches(16, 9)
    for i, ax in enumerate(axes.flat):
    
        dataset_idx = samples[i]
        
        dat = PLOT_DATA[dataset_idx].to(device).unsqueeze(0)
    
        with torch.no_grad():
            outputs = model.reconstruction(dat)
        mu = outputs["px"].mu.view(-1).detach().cpu().numpy()
        sigma = outputs["px"].sigma.view(-1).detach().cpu().numpy()

        probs = outputs["px"].log_prob(dat.view(-1)).exp().detach().cpu().numpy()
        anom_quantile = stats.norm.ppf(1 - ANOMALY_THRESHOLD/2)
        
        #idx = probs < ANOMALY_THRESHOLD

        idx = (dat.view(-1).detach().cpu().numpy()>(mu+anom_quantile*sigma))|(dat.reshape(-1).detach().cpu().numpy()<(mu-anom_quantile*sigma)) 

        anom = np.arange(len(PLOT_DATA[dataset_idx]))[idx.squeeze()]
        
        ax.plot(dat.view(-1).detach().cpu().numpy(), c="b", linewidth=2,label="data")

        ax.plot(mu, c="r", linewidth=2,label = r"Mean reconstruction $\mu$")
        ax.fill_between(list(range(len(mu))), mu-anom_quantile*sigma, mu+anom_quantile*sigma, facecolor='red', alpha=0.3,label =r"$\mu \pm \Phi^{-1}(1-\alpha/2)\cdot \sigma$")        
        ax.scatter(anom, dat.view(-1).detach().cpu().numpy()[anom], c="g", s=50, label="Anomaly")
        ax.set_title(f"Sample {samples[i]}", fontdict={"size": 12})
        #ax.set_ylim(min_val-2*sigma.max(), max_val+2*sigma.max())
        plt.setp(ax.get_xticklabels(), fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels,bbox_to_anchor=(1.15, 0.5))
    fig.tight_layout()
    plt.show()


def total_reconstruction_err(model, dataset, plot=True):
    with torch.no_grad():
        model.eval()
        model_output = model.reconstruction(dataset)
    squared_errors = torch.pow(model_output["px"].mu - dataset.squeeze(), 2) 
    sse_samples = squared_errors.sum(-1).detach().cpu().numpy()
    
    if plot:
        fig, ax = plt.subplots()
        ax.hist(sse_samples, log=True, bins=25)
        ax.set_title("Histogram of Summed Squared Error for each sequence")
        plt.show()
    
    return sse_samples


def outlier_heuristic(model, data, window_size, num_outliers, ANOMALY_THRESHOLD):
    model.eval()
    outliers_idx = []

    # Predict on all sequences and get probabilities
    num_sequences = len(data)
    with torch.no_grad():
        outputs = model.reconstruction(data)
    #probs = outputs["px"].log_prob(train_dataset_simple[:10].to(device).squeeze(-1)).exp().detach().cpu().numpy()
    mu = outputs["px"].mu.detach().cpu().numpy()
    sigma = outputs["px"].sigma.detach().cpu().numpy()
    anom_quantile = stats.norm.ppf(1 - ANOMALY_THRESHOLD/2)
    out = data.view(num_sequences,-1).detach().cpu().numpy()
    indices = (out < mu-anom_quantile*sigma)|(out > mu+anom_quantile*sigma)
    # loop through sequences and detect where prob below ANOMALY THRESHOLD
    for i in range(num_sequences):
        #prob_i = probs[i]
        idx = indices[i] #prob_i < ANOMALY_THRESHOLD
        # SLIDE OVER ALL IDX WHERE PROB < ANOMALY_THRESHOLD
        for j in range(len(idx) - window_size-1):
            # GET WINDOW, IF WINDOW HAS MORE THAN num_outliers, THEN SEQUENCE IS AN OUTLIER!
            window_sequence = idx[j:j+window_size]
            if sum(window_sequence) > num_outliers:
                outliers_idx.append(i)
                break

    return outliers_idx