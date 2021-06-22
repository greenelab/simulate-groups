import numpy as np
from simulate_groups import simulate_ll

source_x, source_y, info_dict = simulate_ll(n=1000, p=100, uncorr_frac=.2,
    num_groups=10, group_sparsity=0.6, seed=139)
print(info_dict['groups'])
print(type(info_dict['groups']))


np.savetxt("source_x.tsv", source_x, delimiter="\t", fmt='%.4f')
np.savetxt("source_y.tsv", source_y, fmt='%i')

source_betas = info_dict['betas']
source_groups_to_keep = info_dict['groups_to_keep']

target_x, target_y, info_dict = simulate_ll(n=700, p=100, uncorr_frac=.2,
    num_groups=10, group_sparsity=0.4, seed=1016, prev_betas=source_betas,
    prev_groups=source_groups_to_keep)
np.savetxt("target_x.tsv", target_x, delimiter="\t", fmt='%.4f')
np.savetxt("target_y.tsv", target_y, fmt='%i')
