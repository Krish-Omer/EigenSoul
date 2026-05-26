from evaluation.gating import gate

# obvious non-duplicate
print(gate(phash_dist=25, resnet_sim=0.90))  # REJECT

# obvious duplicate
print(gate(phash_dist=5, resnet_sim=0.88))   # ACCEPT

# obvious non-duplicate
print(gate(phash_dist=3, resnet_sim=0.40))   # REJECT

# borderline case
print(gate(phash_dist=6, resnet_sim=0.65))   # AMBIGUOUS
