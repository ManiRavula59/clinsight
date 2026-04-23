import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = ['NDCG@10', 'Recall@10', 'MAP@10']
existing_work = [0.484, 0.658, 0.392]
proposed_clinsight = [0.535, 0.665, 0.495]
gains = ['+10.5%', '+1.1%', '+26.3%']

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.35

rects1 = ax.bar(x - width/2, existing_work, width, label='Existing Work (SOTA)', color='#4A5568', alpha=0.8)
rects2 = ax.bar(x + width/2, proposed_clinsight, width, label='Proposed Clinsight', color='#48BB78')

ax.set_ylabel('Score')
ax.set_title('Performance Comparison: Existing Work vs. Proposed Clinsight', fontsize=14, pad=20, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

def autolabel(rects, is_proposed=False):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
        if is_proposed:
            ax.annotate(gains[i], xy=(rect.get_x() + rect.get_width() / 2, height + 0.05),
                        ha='center', va='bottom', color='#48BB78', fontweight='bold', fontsize=11)

autolabel(rects1)
autolabel(rects2, is_proposed=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(0, 0.8)
plt.tight_layout()

plt.savefig('app/data/head_to_head_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Visualization updated with professional labels.")
