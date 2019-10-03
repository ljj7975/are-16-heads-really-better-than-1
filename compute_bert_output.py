import argparse
import csv
import os
import torch
import pandas as pd

from tqdm import tqdm

from BERT.pytorch_pretrained_bert import BertModel, BertForSequenceClassification, BertTokenizer
from BERT import pruning
from utils import chunk, SingleInputBundle, hook_bert_layer_attn, BundleAveragingHook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, required=True)
    parser.add_argument('--task', '-t', type=str, required=True)
    parser.add_argument('--output_folder_name', '-o', type=str, default="bert_output")
    parser.add_argument('--bert-model', type=str, default='bert-base-uncased')
    parser.add_argument('--batch-size', '-b', type=int, default=16)
    parser.add_argument('--finetuned_model', '-f', type=str)
    parser.add_argument('--layers', type=int, nargs='+', default=list(range(12)))
    parser.add_argument(
        "--attention_mask_heads", default="", type=str, nargs="*",
        help="[layer]:[head1],[head2]..."
    )
    parser.add_argument(
        '--reverse_head_mask', action='store_true',
        help="Mask all heads except those specified by `--attention-mask-heads`"
    )
    parser.add_argument(
        "--actually_prune", action='store_true',
        help="Really prune (like, for real)"
    )
    args = parser.parse_args()

    # prepare output dir

    output_dir = os.path.join(args.output_folder_name, args.task)
    if not os.path.exists(args.output_folder_name):
        os.makedirs(output_dir)

    # prepare model

    if args.finetuned_model is None:
        model = BertModel.from_pretrained(args.bert_model)
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model)
        state_dict = torch.load(args.finetuned_model)
        model.load_state_dict(state_dict)
        model = model.bert

    model.cuda()
    model.eval()

    # TODO:: must be dynamic to pruning
    num_attention_heads=model.encoder.layer[0].attention.self.n_heads

    print(f'TASK - {args.task}')
    print(f'layers - {args.layers}')
    print(f'num heads - {num_attention_heads}')

    # Parse pruning descriptor
    to_prune = pruning.parse_head_pruning_descriptors(
        args.attention_mask_heads,
        reverse_descriptors=args.reverse_head_mask,
    )

    print(f'masked heads - {args.attention_mask_heads}')

    # Mask heads
    if args.actually_prune:
        model.prune_heads(to_prune)
    else:
        model.mask_heads(to_prune)

    # hooks
    hooks = []
    for idx in args.layers:
        attn_hook = BundleAveragingHook()
        hook_bert_layer_attn(model, idx, attn_hook)
        hooks.append((f'attn-l{idx}', attn_hook))

    # load data

    data_file = os.path.join(args.data_dir, args.task, "test.tsv")

    df = pd.read_csv(data_file, sep='\t', quoting=3)

    print("total size :", len(df))

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # processing input data

    token_file = open(os.path.join(output_dir, 'tokens.tsv'), 'wt')
    tsv_writer = csv.writer(token_file, delimiter='\t')

    for data in tqdm(list(chunk(list(df.itertuples()), args.batch_size))):

        if "CoLA" in data_file:
            _, _, sentences = list(zip(*data))
        elif "SST-2" in data_file:
            _, _, sentences = list(zip(*data))
        else:
            raise Exception('Unhandled dataset')

        lower_sentences = [ ' '.join(tokenizer.tokenize(s.lower())) for s in sentences ]

        bundle = SingleInputBundle(list(map(str.split, lower_sentences)), tokenizer.vocab)
        bundle.cuda()
        for _, hook in hooks:
            hook.bundle = bundle
        with torch.no_grad():
            model(bundle.token_ids, bundle.segment_ids, bundle.input_mask)

        for padded_sentence in bundle.padded_sentences:
            tsv_writer.writerow(padded_sentence)

    # storing activations

    for name, hook in hooks:
        combined = torch.cat(hook.data)
        print(name, combined.shape)

        torch.save(combined, os.path.join(output_dir, f'{name}.pt'))
        new_shape = combined.size()[:-1] + (num_attention_heads, -1)

        combined = combined.view(*new_shape).permute((1, 0, 2))
        for subhead_idx, subhead_data in enumerate(combined):
            torch.save(subhead_data, os.path.join(output_dir, f'{name}-{subhead_idx}.pt'))
            print(f'\t{name}-{subhead_idx}', subhead_data.shape)


if __name__ == '__main__':
    main()
