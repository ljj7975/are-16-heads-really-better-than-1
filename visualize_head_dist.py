import argparse
import os
import torch

import numpy as np

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
from xlwt import Workbook 
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', required=True)
    parser.add_argument('--method', '-m', type=str, default='pca', choices=['tsne', 'pca', 'lda'])
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--reduction', type=float, default=1.0)
    parser.add_argument('--input_folder_name', '-i', type=str, default="bert_output")
    parser.add_argument('--output_folder_name', '-o', type=str, default="head_distributions")
    args = parser.parse_args()

    np.random.seed(10)
    num_heads = 12

    wb = Workbook()

    # prepare output dir

    output_dir = os.path.join(args.output_folder_name, args.task)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_dir = os.path.join(args.input_folder_name, args.task)

    # one layer at a time
    
    for layer in tqdm(range(args.num_layers)):
        sheet = wb.add_sheet(f'Layer {layer}')

        sheet.write(0, 1, 'head_ind')
        sheet.write(0, 2, 'explained_variance_1')
        sheet.write(0, 3, 'explained_variance_2')
        sheet.write(0, 4, 'explained_variance_ratio_1')
        sheet.write(0, 5, 'explained_variance_ratio_2')
        sheet.write(0, 6, 'singular_values_1')
        sheet.write(0, 7, 'singular_values_2')
        sheet.write(0, 8, 'noise_variance_')

        data = []
        labels = np.arange(0, num_heads)

        label_size = []

        for head_ind in labels:
            head_file = os.path.join(input_dir, f'attn-l{layer}-{head_ind}.pt')
            head_data = torch.load(head_file).numpy()

            num_entry = head_data.shape[0]

            if args.reduction < 1:
                num_selected = round(num_entry*args.reduction)
                random_idx = np.random.randint(num_entry, size=num_selected)
                head_data = head_data[random_idx, :]
                label_size.append(num_selected)
            else:
                label_size.append(num_entry)

            data.append(head_data)

            pca = PCA(n_components=2)
            pca.fit_transform(head_data)

            sheet.write(int(head_ind+1), 1, int(head_ind))
            sheet.write(int(head_ind+1), 2, float(pca.explained_variance_[0]))
            sheet.write(int(head_ind+1), 3, float(pca.explained_variance_[1]))
            sheet.write(int(head_ind+1), 4, float(pca.explained_variance_ratio_[0]))
            sheet.write(int(head_ind+1), 5, float(pca.explained_variance_ratio_[1]))
            sheet.write(int(head_ind+1), 6, float(pca.singular_values_[0]))
            sheet.write(int(head_ind+1), 7, float(pca.singular_values_[1]))
            sheet.write(int(head_ind+1), 8, float(pca.noise_variance_))

        data = np.concatenate(data)
        labels = np.repeat(labels, label_size)

        if args.method == 'pca':
            data = PCA(n_components=2).fit_transform(data)
        elif args.method == 'tsne':
            data = TSNE(n_jobs=24, verbose=True).fit_transform(data)
        elif args.method == 'lda':
            data = LDA(n_components=2).fit_transform(data, labels)

        # plot distribution

        fig, ax = plt.subplots()

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2,
                         box.width, box.height * 0.85])

        color_dict = {
            0 :'r',
            1 :'g',
            2 :'b',
            3 :'y',
            4 :'cyan',
            5 :'magenta',
            6 :'black',
            7 :'grey',
            8 :'orange',
            9 :'purple',
            10 :'navy',
            11 :'pink'
        }

        for head_label in np.arange(0, num_heads):
            ix = np.where(labels == head_label)
            scatter = ax.scatter(data[:,0][ix], data[:,1][ix], c=color_dict[head_label], label=head_label, s=1)

        lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), title="Heads", ncol=int(num_heads/2), prop={'size': 10})
        for handle in lgd.legendHandles:
            handle.set_sizes([100.0])

        ax.set_title(f'Heads distribution for layer {layer}')

        fig.savefig(os.path.join(output_dir, f'attn-l{layer}-heads.png'))

    wb.save(os.path.join(output_dir, "pca_analysis.xls"))


if __name__ == '__main__':
    main()
