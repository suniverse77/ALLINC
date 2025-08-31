import os
import matplotlib.pyplot as plt
from collections import Counter
from cifar10_LT import make_cifar10_lt

imb_ratio = 200

train_data = make_cifar10_lt(
    root = '/home/suno3534/data/_datasets/',
    imb_ratio = imb_ratio, 
    is_train = True, 
    download = True, 
)

os.makedirs('images', exist_ok=True)

labels = [label for _, label in train_data]

counts = Counter(labels)
classes = sorted(counts.keys())
nums = [counts[c] for c in classes]

fig, ax = plt.subplots(figsize=(8,4))
bars = ax.bar(classes, nums)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f'{int(height)}',
        ha='center',
        va='bottom',
        fontsize=8
    )

ax.set_xlabel('Class number')
ax.set_ylabel('# of samples')
ax.set_title(f'Class distribution (IR={imb_ratio})')
ax.set_xticks(classes)
plt.tight_layout()

save_path = os.path.join('images', f'class_distribution_{imb_ratio}.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()