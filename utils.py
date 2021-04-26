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

def add_voigt_sequences(sequences, w, plot=True, seed=None):
    
    if seed:
        np.random.seed(seed)
        
    if w.max() < 1000:
        w_short = True
    else:
        w_short = False
    
    min_uniform = 0.15
    mu = np.random.choice(np.arange(100,900),size=1)
    if mu>800:
        max_uniform = min_uniform
    elif mu < (360+mu*(400-300)/(715-270)):
        max_uniform = 1+mu * (0.5-0.85)/(715-270)
    else:
        max_uniform = 1.1+mu * (0.3-0.8)/(715-270)
        
    voigt_added_sequences = torch.zeros_like(sequences)
    
    for i in range(len(sequences)):
        
        if w_short:
            dat = sequences[i]
        else:
            dat = sequences[i, w<=1000]
        max_val = dat.max() - dat.min()
        mu = np.random.choice(np.arange(100,900),size=1)
        if mu>800:
            max_uniform = min_uniform
        elif mu < (360+mu*(400-300)/(715-270)):
            max_uniform = 1+mu * (0.5-0.85)/(715-270)
        else:
            max_uniform = 1.1+mu * (0.3-0.8)/(715-270)
        alpha = np.random.uniform(low=min_uniform,high=max_uniform,size=1)
        
        if w_short:
            voigt_added_sequences[i] = dat + voigt(w, mu, 12, 12, max_val*alpha)
        else:
            voigt_added_sequences[i] = dat + voigt(w[w<1001], mu, 12, 12, max_val*alpha)
        
    if plot:
        i = np.random.choice(sequences.shape[0],size=16)
        #i = np.arange(16)
        min_height = 0.1
        fig,axes = plt.subplots(4,4,figsize=(20,20))
        for k,ax in enumerate(axes.flat):
            
            if w_short:
                ax.plot(w,voigt_added_sequences[i[k]],label="voigt added",c = 'r')
                ax.plot(w,sequences[i[k]],label="normal",c='b')
            else:
                ax.plot(w[w<1001],voigt_added_sequences[i[k]],label="voigt added",c = 'r')
                ax.plot(w[w<1001],sequences[i[k],w<=1000],label="normal",c='b')
            ax.legend()
            ax.set_title("Testing voigt profile on data for seq:{}".format(i[k]))
        fig.tight_layout()
    
    # ensure we get sequences of shape (BATCH SIZE, SEQ LEN, NUM_FEATURES)
    if len(voigt_added_sequences.shape) < 3:
        return voigt_added_sequences.unsqueeze(-1)
    else:
        return voigt_added_sequences

