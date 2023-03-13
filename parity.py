import os, argparse
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import random
import matplotlib.pyplot as plt
import numpy as np

from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def data_preparation(args, seed_id=42, test_samples=100, test_batchsize=100, return_raw=False):
    # Data generation - different seed each time
    _data = parity(args.n, args.k, args.N, seed=seed_id*17)
    train_dataset = TensorDataset(_data[0], _data[1])
    train_dataloader = DataLoader(train_dataset, batch_size=args.B, shuffle=True)

    data = parity(args.n, args.k, test_samples, seed=2001) # constant test samples
    test_dataset = TensorDataset(data[0], data[1])
    test_dataloader = DataLoader(test_dataset, batch_size=test_batchsize, shuffle=True)

    if (return_raw):
        return train_dataloader, train_dataloader, _data[0][:100], data[0]
    return train_dataloader, test_dataloader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=40, type=int, help='string dimension')
    parser.add_argument('--k', default=3, type=int, help='parity dimension')
    parser.add_argument('--N', default=1000, type=int, help='number of training samples')
    parser.add_argument('--B', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--weight-decay', default=0.01, type=float, help='value of weight decay')
    parser.add_argument('--width', default=1000, type=int, help='width of network')
    parser.add_argument('--n_seeds', default=5, type=int, help='number of random seeds')
    parser.add_argument('--train', action='store_true', help='train models')
    parser.add_argument('--ind-norms', action='store_true', help='plot individual neurons norm')
    parser.add_argument('--global-sparsity', action='store_true', help='compute and plot global sparsity of network over time')
    parser.add_argument('--subnetworks', action='store_true', help='compute and plot subnetworks quantities')
    parser.add_argument('--sparsity-sampling', default=10, help='every how many epochs we compute the global sparsity of the network')
    parser.add_argument('--faithfulness', action='store_true', help='compute faithfulness instead of accuracy for subnetworks')
    parser.add_argument('--lottery-ticket', action='store_true', help='calculate performance of the sparse subnetwork trained from scratch')
    
    return parser.parse_args()

def main():
    args = get_args()
    loss_fn = MyHingeLoss()

    base_dir = f'./results/n_{args.n}/k_{args.k}/N_{args.N}/lr_{args.lr}/wd_{args.weight_decay}/width_{args.width}'
    os.makedirs(base_dir, exist_ok=True)

    fig_path = os.path.join(base_dir, 'figures')
    os.makedirs(fig_path, exist_ok=True)
    
    losses, accs, normss = {'train': [], 'test': []}, {'train': [], 'test': []}, []
    mem_epochs, gen_epochs = [], []
    if args.train:
        for seed_id in range(args.n_seeds):
            torch.manual_seed(seed_id)

            # Data & save_dir preparation
            train_dataloader, test_dataloader = data_preparation(args, seed_id)
            path = os.path.join(base_dir, f'seed{seed_id}_checkpoints')
            os.makedirs(path, exist_ok=True)

            # Model & Optim initialization
            model = FF1(input_dim=args.n, width=args.width)
            model = model.to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            mem_epochs.append(-1)
            gen_epochs.append(-1)
            train_loss, test_loss = [], []
            train_acc, test_acc = [], []
            norms = {'feats':  [],
                    'conx':  []}
            for epoch in range(args.epochs):
                if (epoch % 100 == 0):
                    print(f"Epoch {epoch + 1}\n-------------------------------")

                # Norm statistics
                norms['feats'].append(torch.linalg.norm(list(model.parameters())[0], dim=1).detach().cpu().numpy())
                norms['conx'].append(torch.squeeze(list(model.parameters())[2]).detach().cpu().numpy())

                # Loss & Accuracy statistics
                train_loss.append(loss_calc(train_dataloader, model, loss_fn))
                test_loss.append(loss_calc(test_dataloader, model, loss_fn))

                train_acc.append(acc_calc(train_dataloader, model))
                test_acc.append(acc_calc(test_dataloader, model))

                # Save memorizing / generalizing network
                if (train_acc[-1] > 0.98 and mem_epochs[-1] < 0):
                    print(f'Saving memorizing model - epoch {epoch}')
                    torch.save(model.state_dict(), os.path.join(path, 'memorization.pt'))
                    mem_epochs[-1] = epoch
                # if (test_acc[-1] > 0.98 and gen_epochs[-1] < 0):
                #     print(f'Saving initially generalizing model - epoch {epoch}')
                #     torch.save(model.state_dict(), os.path.join(path, 'initial_generalization.pt'))
                if (epoch == args.epochs - 1):
                    print(f'Saving (final) generalizing model - epoch {epoch}')
                    torch.save(model.state_dict(), os.path.join(path, 'generalization.pt'))
                    gen_epochs[-1] = epoch

                # Save model
                torch.save(model.state_dict(), os.path.join(path, f'model_{epoch}.pt'))
                # Train model
                for id_batch, (x_batch, y_batch) in enumerate(train_dataloader):

                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    pred = model(x_batch)

                    optimizer.zero_grad()
                    loss = loss_fn(pred, y_batch).mean()
                    loss.backward()
                    optimizer.step()
            
            losses['train'].append(train_loss)
            losses['test'].append(test_loss)  

            accs['train'].append(train_acc)
            accs['test'].append(test_acc)

            normss.append(norms)


        # Save files
        with open(os.path.join(base_dir, 'normss'), "wb") as fp:
            pickle.dump(normss, fp)

        with open(os.path.join(base_dir, 'mem_epochs'), "wb") as fp:
            pickle.dump(mem_epochs, fp)

        with open(os.path.join(base_dir, 'gen_epochs'), "wb") as fp:
            pickle.dump(gen_epochs, fp)

        # Save and Plot train (test) curves (acc & loss)
        m1, std1 = mean_and_std_across_seeds(losses['train'])
        np.save(os.path.join(base_dir, 'mean_train_loss'), m1)
        np.save(os.path.join(base_dir, 'std_train_loss'), std1)

        m2, std2 = mean_and_std_across_seeds(losses['test'])
        np.save(os.path.join(base_dir, 'mean_test_loss'), m2)
        np.save(os.path.join(base_dir, 'std_test_loss'), std2)

        m3, std3 = mean_and_std_across_seeds(accs['train'])
        np.save(os.path.join(base_dir, 'mean_train_acc'), m3)
        np.save(os.path.join(base_dir, 'std_train_acc'), std3)

        m4, std4 = mean_and_std_across_seeds(accs['test'])
        np.save(os.path.join(base_dir, 'mean_test_acc'), m4)
        np.save(os.path.join(base_dir, 'std_test_acc'), std4)

        plt.plot(m1, linestyle='-', label='train')
        plt.plot(m2, linestyle='-', label='test')
        plt.fill_between([i for i in range(args.epochs)], m1 - std1, m1 + std1, alpha = 0.3)
        plt.fill_between([i for i in range(args.epochs)], m2 - std2, m2 + std2, alpha = 0.3)
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.xscale('log')
        plt.legend()
        plt.savefig(os.path.join(fig_path, 'loss.pdf'))
        plt.close()

        plt.plot(m3, linestyle='-', label='train')
        plt.plot(m4, linestyle='-', label='test')
        plt.fill_between([i for i in range(args.epochs)], m3 - std3, m3 + std3, alpha = 0.3)
        plt.fill_between([i for i in range(args.epochs)], m4 - std4, m4 + std4, alpha = 0.3)
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.xscale('log')
        plt.legend()
        plt.savefig(os.path.join(fig_path, 'acc.pdf'))
        plt.close()

    else:
        with open(os.path.join(base_dir, 'normss'), "rb") as fp:
            normss = pickle.load(fp)

        with open(os.path.join(base_dir, 'mem_epochs'), "rb") as fp:
            mem_epochs = pickle.load(fp)

        with open(os.path.join(base_dir, 'gen_epochs'), "rb") as fp:
            gen_epochs = pickle.load(fp)


    if args.ind_norms:
        # Show norm evolutions
        for seed_id in range(args.n_seeds):
            best = normss[seed_id]['feats'][gen_epochs[seed_id]].argmax()
            prev_best = normss[seed_id]['feats'][mem_epochs[seed_id]].argmax()

            traj, prev_traj = [], []
            for k in range(args.epochs):
                traj.append(normss[seed_id]['feats'][k][best])
                prev_traj.append(normss[seed_id]['feats'][k][prev_best])

            plt.plot(traj, label='generalizing neuron', lw=2, color='crimson')
            plt.plot(prev_traj, label='memorizing neuron', lw=2, color='navy')
            plt.title(f'Norm evolution of neurons belonging to different circuits - seed {seed_id}')
            plt.xscale('log')
            plt.xlabel('epochs')
            plt.ylabel(r'$\| w \|$')
            plt.legend()
            plt.savefig(os.path.join(fig_path, f'norm_contrast_seed{seed_id}.pdf'))
            plt.show()
            plt.close()

            trajs = []
            for neuron in range(args.width):
                trajs.append([])
                for k in range(args.epochs):
                    trajs[-1].append(normss[seed_id]['feats'][k][neuron])

            for neuron in range(args.width):
                plt.plot(trajs[neuron])
            plt.title(f'Norm evolution of all neurons - seed {seed_id}')
            plt.xlabel('epochs')
            plt.ylabel(r'$\| w \|$')
            plt.xscale('log')
            plt.savefig(os.path.join(fig_path, f'all_neurons_norm_seed{seed_id}.pdf'))
            plt.show()
            plt.close()


    if args.global_sparsity:
        # Global sparsity over time
        sparsities = []
        for seed_id in range(args.n_seeds):
            sparsity = []
            path = os.path.join(base_dir, f'seed{seed_id}_checkpoints')
            train_dataloader, _ = data_preparation(args, seed_id) # load the training dataset of that seed_id
            for epoch in range(0, args.epochs, args.sparsity_sampling):
                _model = FF1(input_dim=args.n, width=args.width).to(device)
                _model.load_state_dict(torch.load(os.path.join(path, f'model_{epoch}.pt')))
                if (epoch < 20):
                    # warm up for irregular behavior in the beginning - non monotonic accuracy (pruning helps apparently?)
                    size, _ = circuit_discovery_linear(epoch, _model, normss[seed_id], train_dataloader, device, args=args)
                else:
                    size, _ = circuit_discovery_binary(epoch, _model, normss[seed_id], train_dataloader, device, args=args)
                    if (size == float('inf')): 
                        # binary search failed?!
                        size, _ = circuit_discovery_linear(epoch, _model, normss[seed_id], train_dataloader, device, args=args)

                sparsity.append(size)
            sparsities.append(sparsity)

        mean_sparsity, std_sparsity = mean_and_std_across_seeds(sparsities)
        np.save(os.path.join(base_dir, 'mean_sparsity_over_time'), mean_sparsity)
        np.save(os.path.join(base_dir, 'std_sparsity_over_time'), std_sparsity)

        plt.plot([i for i in range(0, args.epochs, args.sparsity_sampling)], mean_sparsity, linestyle='-')
        plt.fill_between([i for i in range(0, args.epochs, args.sparsity_sampling)], mean_sparsity - std_sparsity, mean_sparsity + std_sparsity, alpha = 0.3)
        plt.title('Sparsity of network')
        plt.xlabel('epochs')
        plt.ylabel('#neurons')
        plt.xscale('log')
        plt.savefig(os.path.join(fig_path, 'sparsity.pdf'))
        plt.close()


    if args.subnetworks:
        # Subnetworks calculations & reconstruction accuracy
        mem_accs, gen_accs, mem_but_gen_accs, all_but_gen_accs, control_accs = {'train': [], 'test': []}, {'train': [], 'test': []}, {'train': [], 'test': []}, {'train': [], 'test': []}, {'train': [], 'test': []}
        mem_sens, gen_sens, mem_but_gen_sens, all_but_gen_sens, control_sens = {'train': [], 'test': []}, {'train': [], 'test': []}, {'train': [], 'test': []}, {'train': [], 'test': []}, {'train': [], 'test': []}
        mem_norms, gen_norms, mem_but_gen_norms, all_but_gen_norms, control_norms = [], [], [], [], []
        for seed_id in range(args.n_seeds):
            path = os.path.join(base_dir, f'seed{seed_id}_checkpoints')
            train_dataloader, test_dataloader, train_samples, test_samples = data_preparation(args, seed_id, return_raw=True) # load the training dataset of that seed_id

            mem_model = FF1(input_dim=args.n, width=args.width).to(device)
            mem_model.load_state_dict(torch.load(os.path.join(path, f'memorization.pt')))
            mem_size, mem_idx = circuit_discovery_linear(mem_epochs[seed_id], mem_model, normss[seed_id], train_dataloader, device, args=args)
            print(f'Memorizing circuit has size equal to {mem_size} with neurons {mem_idx}')

            gen_model = FF1(input_dim=args.n, width=args.width).to(device)
            gen_model.load_state_dict(torch.load(os.path.join(path, f'generalization.pt')))
            gen_size, gen_idx = circuit_discovery_linear(gen_epochs[seed_id], gen_model, normss[seed_id], train_dataloader, device, args=args)
            print(f'Generalizing (final) circuit has size equal to {gen_size} with neurons {gen_idx}')

            mem_but_gen_idx = list(set(mem_idx) - set(gen_idx))
            all_but_gen_idx = list(set([i for i in range(args.width)]) - set(gen_idx))

            random.seed(seed_id)
            control_idx = random.sample(range(args.width), k=gen_size)

            mem_train_acc, mem_test_acc, gen_train_acc, gen_test_acc, mem_but_gen_train_acc, mem_but_gen_test_acc , all_but_gen_train_acc, all_but_gen_test_acc, control_train_acc, control_test_acc = [], [], [], [], [], [], [], [], [], []
            mem_train_sens, mem_test_sens, gen_train_sens, gen_test_sens, mem_but_gen_train_sens, mem_but_gen_test_sens , all_but_gen_train_sens, all_but_gen_test_sens, control_train_sens, control_test_sens = [], [], [], [], [], [], [], [], [], []
            mem_norm, gen_norm, mem_but_gen_norm, all_but_gen_norm, control_norm = [], [], [], [], []
            # masked inference for the different sets of indices
            for epoch in range(args.epochs):
                _model = FF1(input_dim=args.n, width=args.width).to(device)
                _model.load_state_dict(torch.load(os.path.join(path, f'model_{epoch}.pt')))
                neurons = list(_model.parameters())[0]

                gen_train_acc.append(acc_calc(train_dataloader, _model, gen_idx, device=device, args=args, faithfulness=args.faithfulness))
                gen_test_acc.append(acc_calc(test_dataloader, _model, gen_idx, device=device, args=args, faithfulness=args.faithfulness))
                gen_norm.append(torch.linalg.norm(neurons[gen_idx], dim=1).detach().cpu().numpy().mean())
                gen_train_sens.append(sensitivity_calc(train_samples, _model, gen_idx, device=device, args=args))
                gen_test_sens.append(sensitivity_calc(test_samples, _model, gen_idx, device=device, args=args))

                all_but_gen_train_acc.append(acc_calc(train_dataloader, _model, all_but_gen_idx, device=device, args=args, faithfulness=args.faithfulness))
                all_but_gen_test_acc.append(acc_calc(test_dataloader, _model, all_but_gen_idx, device=device, args=args, faithfulness=args.faithfulness))
                all_but_gen_norm.append(torch.linalg.norm(neurons[all_but_gen_idx], dim=1).detach().cpu().numpy().mean())
                all_but_gen_train_sens.append(sensitivity_calc(train_samples, _model, all_but_gen_idx, device=device, args=args))
                all_but_gen_test_sens.append(sensitivity_calc(test_samples, _model, all_but_gen_idx, device=device, args=args))

                control_train_acc.append(acc_calc(train_dataloader, _model, control_idx, device=device, args=args, faithfulness=args.faithfulness))
                control_test_acc.append(acc_calc(test_dataloader, _model, control_idx, device=device, args=args, faithfulness=args.faithfulness))
                control_norm.append(torch.linalg.norm(neurons[control_idx], dim=1).detach().cpu().numpy().mean())
                control_train_sens.append(sensitivity_calc(train_samples, _model, control_idx, device=device, args=args))
                control_test_sens.append(sensitivity_calc(test_samples, _model, control_idx, device=device, args=args))


            gen_accs['train'].append(gen_train_acc)
            gen_accs['test'].append(gen_test_acc)
            gen_norms.append(gen_norm)
            gen_sens['train'].append(gen_train_sens)
            gen_sens['test'].append(gen_test_sens)

            all_but_gen_accs['train'].append(all_but_gen_train_acc)
            all_but_gen_accs['test'].append(all_but_gen_test_acc)
            all_but_gen_norms.append(all_but_gen_norm)
            all_but_gen_sens['train'].append(all_but_gen_train_sens)
            all_but_gen_sens['test'].append(all_but_gen_test_sens)

            control_accs['train'].append(control_train_acc)
            control_accs['test'].append(control_test_acc)
            control_norms.append(control_norm)
            control_sens['train'].append(control_train_sens)
            control_sens['test'].append(control_test_sens)


        mean_gen_train_acc, std_gen_train_acc = mean_and_std_across_seeds(gen_accs['train'])
        np.save(os.path.join(base_dir, 'mean_gen_train_faith'), mean_gen_train_acc)
        np.save(os.path.join(base_dir, 'std_gen_train_faith'), std_gen_train_acc)

        mean_gen_test_acc, std_gen_test_acc = mean_and_std_across_seeds(gen_accs['test'])
        np.save(os.path.join(base_dir, 'mean_gen_test_faith'), mean_gen_test_acc)
        np.save(os.path.join(base_dir, 'std_gen_test_faith'), std_gen_test_acc)

        mean_all_but_gen_train_acc, std_all_but_gen_train_acc = mean_and_std_across_seeds(all_but_gen_accs['train'])
        np.save(os.path.join(base_dir, 'mean_all_but_gen_train_faith'), mean_all_but_gen_train_acc)
        np.save(os.path.join(base_dir, 'std_all_but_gen_train_faith'), std_all_but_gen_train_acc)

        mean_all_but_gen_test_acc, std_all_but_gen_test_acc = mean_and_std_across_seeds(all_but_gen_accs['test'])
        np.save(os.path.join(base_dir, 'mean_all_but_gen_test_faith'), mean_all_but_gen_test_acc)
        np.save(os.path.join(base_dir, 'std_all_but_gen_test_faith'), std_all_but_gen_test_acc)

        mean_control_train_acc, std_control_train_acc = mean_and_std_across_seeds(control_accs['train'])
        np.save(os.path.join(base_dir, 'mean_control_train_faith'), mean_control_train_acc)
        np.save(os.path.join(base_dir, 'std_control_train_faith'), std_control_train_acc)

        mean_control_test_acc, std_control_test_acc = mean_and_std_across_seeds(control_accs['test'])
        np.save(os.path.join(base_dir, 'mean_control_test_faith'), mean_control_test_acc)
        np.save(os.path.join(base_dir, 'std_control_test_faith'), std_control_test_acc)

        # norms

        mean_gen_norm, std_gen_norm = mean_and_std_across_seeds(gen_norms)
        np.save(os.path.join(base_dir, 'mean_gen_norm'), mean_gen_norm)
        np.save(os.path.join(base_dir, 'std_gen_norm'), std_gen_norm)

        mean_all_but_gen_norm, std_all_but_gen_norm = mean_and_std_across_seeds(all_but_gen_norms)
        np.save(os.path.join(base_dir, 'mean_all_but_gen_norm'), mean_all_but_gen_norm)
        np.save(os.path.join(base_dir, 'std_all_but_gen_norm'), std_all_but_gen_norm)

        mean_control_norm, std_control_norm = mean_and_std_across_seeds(control_norms)
        np.save(os.path.join(base_dir, 'mean_control_norm'), mean_control_norm)
        np.save(os.path.join(base_dir, 'std_control_norm'), std_control_norm)

        # sensitivity

        mean_gen_train_sens, std_gen_train_sens = mean_and_std_across_seeds(gen_sens['train'])
        np.save(os.path.join(base_dir, 'mean_gen_train_sens'), mean_gen_train_sens)
        np.save(os.path.join(base_dir, 'std_gen_train_sens'), std_gen_train_sens)

        mean_gen_test_sens, std_gen_test_sens = mean_and_std_across_seeds(gen_sens['test'])
        np.save(os.path.join(base_dir, 'mean_gen_test_sens'), mean_gen_test_sens)
        np.save(os.path.join(base_dir, 'std_gen_test_sens'), std_gen_test_sens)

        mean_all_but_gen_train_sens, std_all_but_gen_train_sens = mean_and_std_across_seeds(all_but_gen_sens['train'])
        np.save(os.path.join(base_dir, 'mean_all_but_gen_train_sens'), mean_all_but_gen_train_sens)
        np.save(os.path.join(base_dir, 'std_all_but_gen_train_sens'), std_all_but_gen_train_sens)

        mean_all_but_gen_test_sens, std_all_but_gen_test_sens = mean_and_std_across_seeds(all_but_gen_sens['test'])
        np.save(os.path.join(base_dir, 'mean_all_but_gen_test_sens'), mean_all_but_gen_test_sens)
        np.save(os.path.join(base_dir, 'std_all_but_gen_test_sens'), std_all_but_gen_test_sens)

        mean_control_train_sens, std_control_train_sens = mean_and_std_across_seeds(control_sens['train'])
        np.save(os.path.join(base_dir, 'mean_control_train_sens'), mean_control_train_sens)
        np.save(os.path.join(base_dir, 'std_control_train_sens'), std_control_train_sens)

        mean_control_test_sens, std_control_test_sens = mean_and_std_across_seeds(control_sens['test'])
        np.save(os.path.join(base_dir, 'mean_control_test_sens'), mean_control_test_sens)
        np.save(os.path.join(base_dir, 'std_control_test_sens'), std_control_test_sens)


    if args.lottery_ticket:
        for seed_id in range(args.n_seeds):
            path = os.path.join(base_dir, f'seed{seed_id}_checkpoints')
            train_dataloader, test_dataloader = data_preparation(args, seed_id) # load the training dataset of that seed_id

            gen_model = FF1(input_dim=args.n, width=args.width).to(device)
            gen_model.load_state_dict(torch.load(os.path.join(path, f'generalization.pt')))
            gen_size, gen_idx = circuit_discovery_linear(gen_epochs[seed_id], gen_model, normss[seed_id], train_dataloader, device, args=args)
            print(f'Generalizing (final) circuit has size equal to {gen_size} with neurons {gen_idx}')

            # Train from scratch this lottery ticket

            torch.manual_seed(seed_id)
            # Model & Optim initialization
            model = FF1(input_dim=args.n, width=gen_size).to(device)
            init_model = FF1(input_dim=args.n, width=args.width).to(device)
            init_model.load_state_dict(torch.load(os.path.join(path, f'model_0.pt')))
            # pass parameters from initial model
            model.linear1.weight = torch.nn.Parameter(init_model.linear1.weight[gen_idx])
            model.linear1.bias = torch.nn.Parameter(init_model.linear1.bias[gen_idx])
            model.linear2.weight = torch.nn.Parameter(init_model.linear2.weight[:, gen_idx])

            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            train_loss, test_loss = [], []
            train_acc, test_acc = [], []
            norms = {'feats':  [],
                    'conx':  []}
            for epoch in range(args.epochs):
                if (epoch % 100 == 0):
                    print(f"Epoch {epoch + 1}\n-------------------------------")

                # Norm statistics
                norms['feats'].append(torch.linalg.norm(list(model.parameters())[0], dim=1).detach().cpu().numpy())
                norms['conx'].append(torch.squeeze(list(model.parameters())[2]).detach().cpu().numpy())

                # Loss & Accuracy statistics
                train_loss.append(loss_calc(train_dataloader, model, loss_fn))
                test_loss.append(loss_calc(test_dataloader, model, loss_fn))

                train_acc.append(acc_calc(train_dataloader, model))
                test_acc.append(acc_calc(test_dataloader, model))

                # Save model
                torch.save(model.state_dict(), os.path.join(path, f'lottery_model_{epoch}.pt'))
                # Train model
                for id_batch, (x_batch, y_batch) in enumerate(train_dataloader):
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    pred = model(x_batch)

                    optimizer.zero_grad()
                    loss = loss_fn(pred, y_batch).mean()
                    loss.backward()
                    optimizer.step()
            
            losses['train'].append(train_loss)
            losses['test'].append(test_loss)  

            accs['train'].append(train_acc)
            accs['test'].append(test_acc)

            normss.append(norms)

            # Save and Plot train (test) curves (acc & loss)
            m1, std1 = mean_and_std_across_seeds(losses['train'])
            np.save(os.path.join(base_dir, 'lottery_mean_train_loss'), m1)
            np.save(os.path.join(base_dir, 'lottery_std_train_loss'), std1)

            m2, std2 = mean_and_std_across_seeds(losses['test'])
            np.save(os.path.join(base_dir, 'lottery_mean_test_loss'), m2)
            np.save(os.path.join(base_dir, 'lottery_std_test_loss'), std2)

            m3, std3 = mean_and_std_across_seeds(accs['train'])
            np.save(os.path.join(base_dir, 'lottery_mean_train_acc'), m3)
            np.save(os.path.join(base_dir, 'lottery_std_train_acc'), std3)

            m4, std4 = mean_and_std_across_seeds(accs['test'])
            np.save(os.path.join(base_dir, 'lottery_mean_test_acc'), m4)
            np.save(os.path.join(base_dir, 'lottery_std_test_acc'), std4)

            plt.plot(m1, linestyle='-', label='train')
            plt.plot(m2, linestyle='-', label='test')
            plt.fill_between([i for i in range(args.epochs)], m1 - std1, m1 + std1, alpha = 0.3)
            plt.fill_between([i for i in range(args.epochs)], m2 - std2, m2 + std2, alpha = 0.3)
            plt.title('Loss of Lottery Ticket')
            plt.xlabel('epochs')
            plt.xscale('log')
            plt.legend()
            plt.savefig(os.path.join(fig_path, 'lottery_loss.pdf'))
            plt.close()

            plt.plot(m3, linestyle='-', label='train')
            plt.plot(m4, linestyle='-', label='test')
            plt.fill_between([i for i in range(args.epochs)], m3 - std3, m3 + std3, alpha = 0.3)
            plt.fill_between([i for i in range(args.epochs)], m4 - std4, m4 + std4, alpha = 0.3)
            plt.title('Accuracy of Lottery Ticket')
            plt.xlabel('epochs')
            plt.xscale('log')
            plt.legend()
            plt.savefig(os.path.join(fig_path, 'lottery_acc.pdf'))
            plt.close()

if __name__ == '__main__':
    main()