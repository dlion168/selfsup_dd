import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def run(mode, dataloader, net, teacher, optimizer, criterion, device, writer=None, step=0):
    """
    Run one epoch of training or evaluation for distillation.

    Parameters:
        mode (str): "train" or "eval"
        dataloader (DataLoader): DataLoader to iterate over the dataset.
        net (nn.Module): The student network to train or evaluate.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (callable): Loss function.
        device (str): Device to use ("cuda:0", "cpu", etc.).
        teacher (nn.Module): The teacher network used for distillation.
        writer (SummaryWriter, optional): TensorBoard writer for logging metrics.
        step (int, optional): Starting step for logging (updated per batch).
    
    Returns:
        float: Average loss over the epoch.
    """
    total_loss = 0.0
    total_samples = 0
    current_step = step

    # Ensure teacher network is in evaluation mode
    teacher.eval()

    if mode == "train":
        net.train()
        # Wrap dataloader with tqdm for progress bar
        for batch in tqdm(dataloader, desc="Training"):
            # Assume batch is a tuple (inputs, _); the second element is ignored.
            inputs, _ = batch
            inputs = inputs.to(device)

            optimizer.zero_grad()

            # Compute teacher outputs (targets) with no gradient computation.
            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            # Compute student network outputs.
            student_outputs = net(inputs)

            # Compute loss between student outputs and teacher outputs.
            loss = criterion(student_outputs, teacher_outputs)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Log batch-level metrics to TensorBoard if writer is provided
            if writer is not None:
                writer.add_scalar('Loss/Train', loss.item(), current_step)
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], current_step)
                current_step += 1

        avg_loss = total_loss / total_samples

    elif mode == "eval":
        net.eval()
        with torch.no_grad():
            # Wrap dataloader with tqdm for progress bar
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs, _ = batch
                inputs = inputs.to(device)

                teacher_outputs = teacher(inputs)
                student_outputs = net(inputs)
                loss = criterion(student_outputs, teacher_outputs)

                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        
        # Log batch-level metrics to TensorBoard if writer is provided
        if writer is not None:
            writer.add_scalar('Loss/Eval', avg_loss, current_step)


    else:
        raise ValueError("Unsupported mode. Please choose 'train' or 'eval'.")

    return avg_loss

# Example usage:
# train_loss = run("train", dataloader=trainloader, net=student_net, optimizer=student_optim,
#                  criterion=criterion, device=args.device, teacher=teacher_net,
#                  writer=expert_writer, step=global_step)