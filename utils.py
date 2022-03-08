### Check exon boundaries
## "min_seq_len": 2**10,
## "max_seq_len": 2**11,
# trainer = modules.Trainer(param_vals, model, data_path)

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(20, 6))
for i in range(len(trainer.training_dset)):
    tr = trainer.training_dset[i]
    exons = tr[0][4, :].numpy()
    axs[i].plot(np.arange(len(exons)), exons)
    axs[i].set_title(tr[4])
plt.show()

