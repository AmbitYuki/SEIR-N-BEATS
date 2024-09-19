import torch
from matplotlib import pyplot as plt
import seaborn as sns
import os.path
b = torch.load('temp_attention_weight.pth')
b = b.detach().numpy()
b = b.squeeze(0)
print(b)

def attention_plot(attention, x_texts=None, y_texts=None, figsize=(15, 10), annot=False, figure_path='./figures',
                   figure_name='attention_weight1.png'):
    plt.clf()
    # fig, ax = plt.subplots(figsize=figsize,dpi =250)
    plt.figure(figsize=figsize, dpi=250)
    # sns.set(font_scale=1.25)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(attention,
                     cbar=True,
                     # cmap="RdBu_r",
                     cmap="nipy_spectral",
                     annot=annot,
                     square=False,
                     fmt='.2f',
                     annot_kws={'size': 10}

                     )
    plt.xticks(fontsize=13)
    plt.savefig('tem可解释性.svg', bbox_inches='tight', dpi=300, pad_inches=0.0)
    # if os.path.exists(figure_path) is False:
    #     os.makedirs(figure_path)
    # plt.xlabel('The location of hidden state')
    # plt.savefig(os.path.join(figure_path, figure_name))
    # plt.close()
    plt.show()

attention_plot(b, annot=False, figsize=(15, 13), figure_path='./figures', figure_name='attention_weight_tem')