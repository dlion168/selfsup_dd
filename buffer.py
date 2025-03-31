import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data.wrapper import get_pretrain_loader, get_eval_loader
from algorithms.wrapper import get_algorithm
from models.tera import tera
import multiprocessing as mp
from functools import partial

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

def save_checkpoint(model, optimizer, epoch, global_step, save_path, args):
    checkpoint = {
        "Transformer": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args)
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint: {save_path}")

def train_expert(expert_idx, gpu_id, args, save_dir, log_dir):
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(f"Training expert {expert_idx} on {device}...")
    trainloader = get_pretrain_loader(batch_size=args.pre_batch_size)
    evalloader = get_eval_loader(batch_size=args.pre_batch_size)

    criterion = nn.MSELoss().to(device) if args.criterion == "mse" else nn.CrossEntropyLoss().to(device)
    train_algo = get_algorithm("distill_from_model")

    pretrained_net = tera(layers=args.layers).to(device)
    torch.manual_seed(expert_idx)
    if torch.cuda.is_available(): torch.cuda.manual_seed(expert_idx)
    student_net = tera(layers=args.layers, init_model=True).to(device)

    optimizer = torch.optim.SGD(student_net.parameters(), lr=args.lr_teacher, momentum=args.mom, weight_decay=args.l2)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"expert_{expert_idx}"))

    # Save initial checkpoint
    save_checkpoint(student_net, optimizer, epoch=0, global_step=0,
                    save_path=os.path.join(save_dir, f"teacher_expert_{expert_idx}_epoch_0.pt"),
                    args=args)

    global_step = 0
    for epoch in range(1, args.train_epochs + 1):
        train_loss = train_algo.run("train", dataloader=trainloader, net=student_net,
                                    teacher=pretrained_net, optimizer=optimizer,
                                    criterion=criterion, device=device, writer=writer,
                                    step=global_step)
        eval_loss = train_algo.run("eval", dataloader=evalloader, net=student_net,
                                   teacher=pretrained_net, optimizer=optimizer,
                                   criterion=criterion, device=device, writer=writer,
                                   step=global_step)
        print(f"Expert {expert_idx}\tEpoch {epoch}\tTrain Loss {train_loss:.4f}\tEval Loss {eval_loss:.4f}")
        global_step += len(trainloader)

        save_checkpoint(student_net, optimizer, epoch=epoch, global_step=global_step,
                        save_path=os.path.join(save_dir, f"teacher_expert_{expert_idx}_epoch_{epoch}.pt"),
                        args=args)

    writer.close()
    print(f"Expert {expert_idx} training completed.")

def main(args):
    num_gpus = torch.cuda.device_count()
    args.device = 'cpu' if num_gpus == 0 else 'cuda:0'
    print(f"Available GPUs: {num_gpus}\nHyper-parameters: {args.__dict__}")

    save_dir = os.path.join(args.buffer_path, args.exp, args.dataset, args.model)
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, 'logs')

    num_processes = min(args.num_experts, max(1, num_gpus))
    for batch_start in range(0, args.num_experts, num_processes):
        batch_end = min(batch_start + num_processes, args.num_experts)
        gpu_ids = list(range(num_gpus))[:num_processes]
        expert_indices = range(batch_start, batch_end)

        with mp.Pool(processes=num_processes) as pool:
            pool.starmap(partial(train_expert, args=args, save_dir=save_dir, log_dir=log_dir),
                         [(idx, gpu_ids[i]) for i, idx in enumerate(expert_indices)])
        print(f"Completed experts {batch_start} to {batch_end - 1}")

    print("All experts trained successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect Teacher Network Training Trajectories Concurrently')
    parser.add_argument('-e', '--exp', type=str, default='0320')
    parser.add_argument('--dataset', type=str, default='librispeech100')
    parser.add_argument('--model', type=str, default='tera')
    parser.add_argument('--num_experts', type=int, default=10)
    parser.add_argument('--lr_teacher', type=float, default=0.001)
    parser.add_argument('--pre_batch_size', type=int, default=32)
    parser.add_argument('--buffer_path', type=str, default='./buffers')
    parser.add_argument('--train_epochs', type=int, default=30)
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--criterion', type=str, default="mse")
    parser.add_argument('--layers', nargs='+', type=int, default=[4,8,12])
    args = parser.parse_args()
    main(args)


# import os
# import argparse
# import torch
# import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
# from data.wrapper import get_pretrain_loader
# from algorithms.wrapper import get_algorithm
# from models.tera import tera

# def main(args):
#     args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
#     print('Hyper-parameters:\n', args.__dict__)

#     # Set the directory to save the buffers and models
#     save_dir = os.path.join(args.exp, args.buffer_path, args.dataset, args.model)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # Setup TensorBoard SummaryWriter (main log directory)
#     log_dir = os.path.join(save_dir, 'logs')

#     # Load the same dataset as in train.py (pretraining dataset)
#     print("Loading dataset...")
#     trainloader = get_pretrain_loader(batch_size=args.pre_batch_size)
#     print("Dataset loaded.")

#     # Define the loss function
#     if args.criterion == "mse":
#         criterion = nn.MSELoss().to(args.device)
#     elif args.criterion == "ce":
#         criterion = nn.CrossEntropyLoss().to(args.device)
#     else:
#         raise NotImplementedError(f"Loss function {args.criterion} is not supported.")

#     # Get the training algorithm for distillation
#     train_algo = get_algorithm("distill_from_model")
#     # Load the pretrained teacher network (used to generate targets)
#     pretrained_net = tera().to(args.device)
    
#     # Collect teacher network training trajectories
#     # trajectories = []
#     for expert_idx in range(args.num_experts):
#         print(f"\nTraining expert {expert_idx} ...")
#         # Create the student network using the speech model (tera) with random initialization
#         student_net = tera(init_model=True).to(args.device)

#         teacher_optim = torch.optim.SGD(student_net.parameters(), lr=args.lr_teacher, momentum=args.mom, weight_decay=args.l2)
#         teacher_optim.zero_grad()

#         # Setup a TensorBoard writer for this expert
#         expert_log_dir = os.path.join(log_dir, f"expert_{expert_idx}")
#         expert_writer = SummaryWriter(log_dir=expert_log_dir)
        
#         model_save_path = os.path.join(save_dir, f"teacher_expert_{expert_idx}_epoch_0.pt")
#         torch.save(student_net.state_dict(), model_save_path)

#         # Record initial parameters (DEBUG: initial parameter snapshot)
#         # timestamps = []
#         # timestamps.append([p.detach().cpu() for p in student_net.parameters()])
#         print("DEBUG: Initial parameters recorded.")
        
#         # Initialize a global step counter for batches
#         global_step = 0

#         # Train the teacher (student network) and record a parameter snapshot after each epoch
#         for epoch_idx in range(args.train_epochs):
#             train_loss = train_algo.run("train", dataloader=trainloader, net=student_net, teacher=pretrained_net, 
#                                         optimizer=teacher_optim, criterion=criterion, device=args.device,
#                                         writer=expert_writer, step=global_step)
#             print(f"Expert {expert_idx} \t Epoch: {epoch_idx} \t Training Loss: {train_loss}")
            
#             # Update global_step based on the number of batches in the epoch
#             global_step += len(trainloader)
            
#             # Save the teacher (student network) model checkpoint every epoch
#             model_save_path = os.path.join(save_dir, f"teacher_expert_{expert_idx}_epoch_{epoch_idx+1}.pt")
#             torch.save(student_net.state_dict(), model_save_path)
#             print(f"Saved teacher model checkpoint at {model_save_path}")

#             # Record parameter snapshot
#             # timestamps.append([p.detach().cpu() for p in student_net.parameters()])
#             print("DEBUG: Epoch ended, parameter snapshot recorded.")

#         # trajectories.append(timestamps)

#         # Save trajectory checkpoint for this expert after finishing all epochs
#         # expert_trajectory_path = os.path.join(save_dir, f"replay_buffer_expert_{expert_idx}.pt")
#         # print(f"Saving trajectory checkpoint to {expert_trajectory_path}...")
#         # torch.save(timestamps, expert_trajectory_path)
#         expert_writer.close()  # Close the expert's TensorBoard writer


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Collect Teacher Network Training Trajectories')
#     parser.add_argument('-e', '--exp', type=str, default='0320', help='Name of the experiment')
#     parser.add_argument('--dataset', type=str, default='librispeech100', help='Name of the dataset')
#     parser.add_argument('--model', type=str, default='tera', help='Model name')
#     parser.add_argument('--num_experts', type=int, default=10, help='Number of experts (trajectories)')
#     parser.add_argument('--lr_teacher', type=float, default=0.001, help="Teacher network's learning rate")
#     parser.add_argument('--pre_batch_size', type=int, default=32, help='Batch size for pretraining data loader')
#     parser.add_argument('--data_path', type=str, default='/home/data', help='Path to the dataset')
#     parser.add_argument('--buffer_path', type=str, default='./buffers', help='Path to save the buffers and models')
#     parser.add_argument('--train_epochs', type=int, default=20, help='Number of training epochs')
#     parser.add_argument('--mom', type=float, default=0.9, help='Momentum parameter')
#     parser.add_argument('--l2', type=float, default=1e-4, help='L2 regularization factor')
#     parser.add_argument('--criterion', type=str, default="mse", help='Loss function (mse or ce)')
#     parser.add_argument('--device', type=int, default=0, help='GPU id to use')
#     parser.add_argument('--img_size', type=int, default=160000, help='Audio length (number of samples)')

#     args = parser.parse_args()
#     main(args)
