import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

import argparse
import os
import time

from pyHGT.model import *
from pyHGT.data import *
from torch.nn.utils.rnn import pad_sequence

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
        return device, True
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
        return device, True
    else:
        device = torch.device('cpu')
        print("Using CPU")
        return device, False


DEVICE, HAS_ACCELERATION = get_device()
CUDA = HAS_ACCELERATION


def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument("-data", "--data",
                      default="./data/ICEWS14_forecasting", help="data directory")
    args.add_argument('--dataset', type=str, default='ICEWS14_forecasting')
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=100, help="Number of epochs")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-6, help="L2 reglarization for conv")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=200, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-4)

    args = args.parse_args()
    return args


def save_model(model, name, folder_name, epoch):
    print("Saving Model")
    torch.save(model.state_dict(),
               (folder_name + "trained"+str(epoch)+".pth"))
    print("Done saving Model")


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


args = parse_args()

mkdirs('./results/bestmodel/{}/conv'.format(args.dataset))
mkdirs('./results/bestmodel/{}/gat'.format(args.dataset))
model_state_file = './results/bestmodel/{}/'.format(args.dataset)


def load_data(args):
    with open(os.path.join('{}'.format(args.data), 'stat.txt'), 'r') as fr:
        for line in fr:
            line_split = line.split()
            num_e, num_r = int(line_split[0]), int(line_split[1])

    relation_embeddings = np.random.randn(num_r * 2, args.embedding_size)
    print("Initialised relations and entities randomly")
    return num_e, num_r, torch.FloatTensor(relation_embeddings)


num_e, num_r, relation_embeddings = load_data(args)

print("Initial relation dimensions {}".format(relation_embeddings.size()))


def list_to_array(x, pad):
    dff = pd.concat([pd.DataFrame({'{}'.format(index): labels})
                    for index, labels in enumerate(x)], axis=1)
    return dff.fillna(pad).values.T.astype(int)


def train_conv(args):
    print("Defining model")
    model = TypeGAT(num_e, num_r*2, relation_embeddings, args.embedding_size)
    if HAS_ACCELERATION:
        model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.5, last_epoch=-1)

    margin_loss = torch.nn.SoftMarginLoss()
    cosine_loss = nn.CosineEmbeddingLoss(margin=0.0)

    epoch_losses = []
    print("Number of epochs {}".format(args.epochs_conv))

    graph_train = renamed_load(
        open(os.path.join(args.data + '/graph_preprocess_train.pk'), 'rb'))

    total_time_steps = len(graph_train.t_r_id_p_dict)
    print(f"Total time steps per epoch: {total_time_steps}")

    for epoch in range(args.epochs_conv):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch + 1}/{args.epochs_conv}")
        print(f"{'='*50}")

        model.train()
        start_time = time.time()
        epoch_loss = []

        time_steps = list(graph_train.t_r_id_p_dict.items())
        progress_bar = tqdm(
            time_steps, desc=f"Epoch {epoch+1}", unit="timestep")

        batch_count = 0
        loss_sum = 0.0

        for step_idx, (t, r_dict) in enumerate(progress_bar):
            step_start_time = time.time()

            batch_values = []
            batch_paths_id = []
            batch_relation = []
            batch_his_r = []

            # for neg paths
            path_r = []
            path_neg_index = []

            # Count total batches in this time step
            total_relations = len(r_dict)
            relations_processed = 0

            for r, id_p in r_dict.items():
                len_r = 0
                p_neg_temp = []
                for id, ps in id_p.items():
                    len_r = len_r + len(ps)

                    value = [-1] * len(ps)
                    value[0] = 1
                    batch_values.extend(value)
                    batch_paths_id.extend(ps)

                    batch_his_r.extend(graph_train.r_copy[t][r][id])
                    if len(ps) > 1:
                        p_neg_temp.extend(
                            list(eval('['+str(ps[1:]).replace("[", '').replace("]", '')+']')))

                batch_relation.extend([r]*len_r)
                path_r.extend([r]*len(p_neg_temp))
                path_neg_index.extend(p_neg_temp)

                relations_processed += 1

            if len(batch_values) > 0:
                path_values = [-1]*len(path_r)
                path_values = torch.FloatTensor(
                    np.expand_dims(np.array(path_values), axis=1))
                path_neg_index = torch.LongTensor(np.array(path_neg_index))
                path_r = torch.LongTensor(np.array(path_r))

                batch_paths_id = torch.LongTensor(
                    list_to_array(batch_paths_id, 0))
                batch_relation = torch.LongTensor(np.array(batch_relation))
                batch_values = torch.FloatTensor(
                    np.expand_dims(np.array(batch_values), axis=1))
                batch_his_r = torch.LongTensor(
                    list_to_array(batch_his_r, num_r*2))

                paths = graph_train.t_paths[t]
                paths_time = graph_train.t_paths_time[t]
                lengths = graph_train.t_paths_len[t]

                if len(paths) != 0:
                    paths = pad_sequence([torch.LongTensor(np.array(p)) for p in paths], batch_first=True,
                                         padding_value=num_r*2)
                    paths_time = pad_sequence([torch.LongTensor(np.array(p)) for p in paths_time], batch_first=True,
                                              padding_value=0)
                else:
                    paths = torch.LongTensor(np.array(paths))
                    paths_time = torch.LongTensor(np.array(paths_time))
                lengths = torch.LongTensor(np.array(lengths))

                if HAS_ACCELERATION:
                    batch_paths_id = Variable(batch_paths_id).to(DEVICE)
                    batch_relation = Variable(batch_relation).to(DEVICE)
                    batch_his_r = Variable(batch_his_r).to(DEVICE)
                    paths = Variable(paths).to(DEVICE)
                    paths_time = Variable(paths_time).to(DEVICE)
                    lengths = Variable(lengths).to(DEVICE)
                    path_r = Variable(path_r).to(DEVICE)
                    path_neg_index = Variable(path_neg_index).to(DEVICE)
                    batch_values = Variable(batch_values).to(DEVICE)
                    path_values = Variable(path_values).to(DEVICE)
                else:
                    batch_paths_id = Variable(batch_paths_id)
                    batch_relation = Variable(batch_relation)
                    batch_his_r = Variable(batch_his_r)
                    paths = Variable(paths)
                    paths_time = Variable(paths_time)
                    lengths = Variable(lengths)
                    path_r = Variable(path_r)
                    path_neg_index = Variable(path_neg_index)
                    batch_values = Variable(batch_values)
                    path_values = Variable(path_values)

                optimizer.zero_grad()

                try:
                    preds, p_emb, r_emb = model.forward2(batch_paths_id, batch_relation, paths, paths_time, lengths, path_r,
                                                         path_neg_index, batch_his_r)

                    del batch_paths_id, batch_relation, paths, paths_time, lengths, path_r, path_neg_index

                    loss_e = margin_loss(preds.view(-1), batch_values.view(-1))
                    loss_f = cosine_loss(p_emb, r_emb, path_values.view(-1))
                    del preds, p_emb, r_emb

                    loss = loss_e  # You can add loss_f back if needed

                    if torch.isnan(loss):
                        print(
                            f"Warning: NaN loss detected at timestep {t}, skipping...")
                        continue

                    loss.backward()
                    optimizer.step()

                    batch_count += 1
                    current_loss = loss.item()
                    epoch_loss.append(current_loss)
                    loss_sum += current_loss

                    avg_loss = loss_sum / batch_count
                    step_time = time.time() - step_start_time

                    progress_bar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'Avg_Loss': f'{avg_loss:.4f}',
                        'Step_Time': f'{step_time:.2f}s',
                        'Relations': f'{relations_processed}/{total_relations}'
                    })

                except Exception as e:
                    print(f"Error in forward pass at timestep {t}: {e}")
                    continue

            if (step_idx + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                avg_loss = loss_sum / max(batch_count, 1)
                print(f"\n  Step {step_idx + 1}/{total_time_steps} | "
                      f"Avg Loss: {avg_loss:.4f} | "
                      f"Batches: {batch_count} | "
                      f"Elapsed: {elapsed_time:.1f}s")

        progress_bar.close()

        scheduler.step()

        epoch_time = time.time() - start_time
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0

        print(f"\n{'='*50}")
        print(f"EPOCH {epoch + 1} SUMMARY:")
        print(f"  Average Loss: {avg_epoch_loss:.6f}")
        print(f"  Total Batches: {batch_count}")
        print(f"  Epoch Time: {epoch_time:.1f}s")
        print(f"  Batches/sec: {batch_count/epoch_time:.2f}")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*50}")

        epoch_losses.append(avg_epoch_loss)

        save_model(model, args.data, model_state_file, epoch)

        if len(epoch_losses) > 5:
            recent_losses = epoch_losses[-5:]
            if all(abs(recent_losses[i] - recent_losses[i-1]) < 1e-6 for i in range(1, len(recent_losses))):
                print("Early stopping: Loss has converged")
                break

    print("\nTraining completed!")
    print(f"Final loss: {epoch_losses[-1]:.6f}")
    return epoch_losses


def evaluate_conv(args):
    model = TypeGAT(num_e, num_r*2, relation_embeddings, args.embedding_size)
    model.load_state_dict(torch.load(
        '{0}/trained99.pth'.format(model_state_file), map_location=DEVICE), strict=False)

    if HAS_ACCELERATION:
        model.to(DEVICE)

    model.eval()

    with torch.no_grad():
        mr, mrr, hits1, hits3, hits10, hits100 = 0, 0, 0, 0, 0, 0
        test_size = 0
        graph_test = renamed_load(
            open(os.path.join(args.data + '/graph_preprocess_test.pk'), 'rb'))
        for t, r_dict in graph_test.t_r_id_p_dict.items():
            size = 0
            ranks_tail = []
            reciprocal_ranks_tail = []
            hits_at_100_tail = 0
            hits_at_ten_tail = 0
            hits_at_three_tail = 0
            hits_at_one_tail = 0

            for r, id_p in r_dict.items():
                for id, ps in id_p.items():
                    len_r = 0
                    batch_paths_id = []
                    batch_relation = []

                    batch_his_r = []

                    size = size + 1
                    len_r = len_r + len(ps)
                    batch_paths_id.extend(ps)

                    batch_relation.extend([r] * len_r)

                    batch_his_r.extend(graph_test.r_copy[t][r][id])

                    batch_paths_id = torch.LongTensor(
                        list_to_array(batch_paths_id, 0))
                    batch_relation = torch.LongTensor(np.array(batch_relation))

                    batch_his_r = torch.LongTensor(
                        list_to_array(batch_his_r, num_r * 2))

                    paths = graph_test.t_paths[t]
                    lengths = graph_test.t_paths_len[t]
                    paths_time = graph_test.t_paths_time[t]

                    if len(paths) != 0:
                        paths = pad_sequence([torch.LongTensor(np.array(p)) for p in paths], batch_first=True,
                                             padding_value=num_r*2)
                        paths_time = pad_sequence([torch.LongTensor(np.array(p)) for p in paths_time], batch_first=True,
                                                  padding_value=0)
                    else:
                        paths = torch.LongTensor(np.array(paths))
                        paths_time = torch.LongTensor(np.array(paths_time))
                    lengths = torch.LongTensor(np.array(lengths))

                    if HAS_ACCELERATION:
                        batch_paths_id = Variable(batch_paths_id).to(DEVICE)
                        batch_relation = Variable(batch_relation).to(DEVICE)
                        paths = Variable(paths).to(DEVICE)
                        lengths = Variable(lengths).to(DEVICE)
                        paths_time = Variable(paths_time).to(DEVICE)
                        batch_his_r = Variable(batch_his_r).to(DEVICE)
                    else:
                        batch_paths_id = Variable(batch_paths_id)
                        batch_relation = Variable(batch_relation)
                        paths = Variable(paths)
                        lengths = Variable(lengths)
                        paths_time = Variable(paths_time)
                        batch_his_r = Variable(torch.LongTensor(batch_his_r))

                    scores_tail = model.test(
                        batch_paths_id, batch_relation, paths, lengths, paths_time, batch_his_r)

                    del batch_paths_id, batch_relation, paths, lengths

                    sorted_scores_tail, sorted_indices_tail = torch.sort(
                        scores_tail.view(-1), dim=-1, descending=True)
                    del scores_tail

                    # Just search for zeroth index in the sorted scores, we appended valid triple at top
                    ranks_tail.append(
                        np.where(sorted_indices_tail.cpu().numpy() == 0)[0][0] + 1)
                    reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])

            for i in range(len(ranks_tail)):
                if ranks_tail[i] <= 100:
                    hits_at_100_tail = hits_at_100_tail + 1
                if ranks_tail[i] <= 10:
                    hits_at_ten_tail = hits_at_ten_tail + 1
                if ranks_tail[i] <= 3:
                    hits_at_three_tail = hits_at_three_tail + 1
                if ranks_tail[i] == 1:
                    hits_at_one_tail = hits_at_one_tail + 1

            assert len(ranks_tail) == len(reciprocal_ranks_tail)
            if len(ranks_tail) == 0:
                continue

            t_hits100 = hits_at_100_tail / len(ranks_tail)
            t_hits10 = hits_at_ten_tail / len(ranks_tail)
            t_hits3 = hits_at_three_tail / len(ranks_tail)
            t_hits1 = hits_at_one_tail / len(ranks_tail)
            t_mr = sum(ranks_tail) / len(ranks_tail)
            t_mrr = sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)

            print("\nCumulative stats are -> ")
            print("Hits@100 are {}".format(t_hits100))
            print("Hits@10 are {}".format(t_hits10))
            print("Hits@3 are {}".format(t_hits3))
            print("Hits@1 are {}".format(t_hits1))
            print("Mean rank {}".format(t_mr))
            print("Mean Reciprocal Rank {}".format(t_mrr))

            test_size = test_size + size

            mrr += t_mrr * size
            mr += t_mr * size
            hits1 += t_hits1 * size
            hits3 += t_hits3 * size
            hits10 += t_hits10 * size
            hits100 += t_hits100 * size

        mrr = mrr / test_size
        mr = mr / test_size
        hits1 = hits1 / test_size
        hits3 = hits3 / test_size
        hits10 = hits10 / test_size
        hits100 = hits100 / test_size

        print("MR : {:.6f}".format(mr))
        print("MRR : {:.6f}".format(mrr))
        print("Hits @ 1: {:.6f}".format(hits1))
        print("Hits @ 3: {:.6f}".format(hits3))
        print("Hits @ 10: {:.6f}".format(hits10))
        print("Hits @ 100: {:.6f}".format(hits100))


train_conv(args)
evaluate_conv(args)
