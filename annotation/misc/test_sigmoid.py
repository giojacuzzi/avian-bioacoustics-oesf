import numpy as np

# Logit confidence values
logits = [
    0.07, 
    0.58,
    0.97,
    2.61,
    3.43
]

for logit in logits:
    sigmoid = 1 / (1 + np.exp(-logit))

    sensitivity = -1
    birdnet_sigmoid = 1 / (1.0 + np.exp(sensitivity * np.clip(logit, -15, 15)))

    print(f'logit {logit}')
    print(f'sigmoid {sigmoid} (standard)')
    print(f'sigmoid {birdnet_sigmoid} (birdnet)')
