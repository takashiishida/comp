import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def assump_free_loss(f, K, labels, ccp):
    """Assumption free loss (based on Thm 1) is equivalent to non_negative_loss if the max operator's threshold is negative inf."""
    return non_negative_loss(f=f, K=K, labels=labels, ccp=ccp, beta=np.inf)

def non_negative_loss(f, K, labels, ccp, beta):
    ccp = torch.from_numpy(ccp).float().to(device)
    neglog = -F.log_softmax(f, dim=1)
    loss_vector = torch.zeros(K, requires_grad=True).to(device)
    temp_loss_vector = torch.zeros(K).to(device)
    for k in range(K):
        idx = labels == k
        if torch.sum(idx).item() > 0:
            idxs = idx.byte().view(-1,1).repeat(1,K)
            neglog_k = torch.masked_select(neglog, idxs).view(-1,K)
            temp_loss_vector[k] = -(K-1) * ccp[k] * torch.mean(neglog_k, dim=0)[k]  # average of k-th class loss for k-th comp class samples
            loss_vector = loss_vector + torch.mul(ccp, torch.mean(neglog_k, dim=0))  # only k-th in the summation of the second term inside max 
    loss_vector = loss_vector + temp_loss_vector
    count = np.bincount(labels.data.cpu()).astype('float')
    while len(count) < K:
        count = np.append(count, 0) # when largest label is below K, bincount will not take care of them
    loss_vector_with_zeros = torch.cat((loss_vector.view(-1,1), torch.zeros(K, requires_grad=True).view(-1,1).to(device)-beta), 1)
    max_loss_vector, _ = torch.max(loss_vector_with_zeros, dim=1)
    final_loss = torch.sum(max_loss_vector)
    return final_loss, torch.mul(torch.from_numpy(count).float().to(device), loss_vector)

def forward_loss(f, K, labels):
    Q = torch.ones(K,K) * 1/(K-1)
    Q = Q.to(device)
    for k in range(K):
        Q[k,k] = 0
    q = torch.mm(F.softmax(f, 1), Q)
    return F.nll_loss(q.log(), labels.long())

def pc_loss(f, K, labels):
    sigmoid = nn.Sigmoid()
    fbar = f.gather(1, labels.long().view(-1, 1)).repeat(1, K)
    loss_matrix = sigmoid( -1. * (f - fbar)) # multiply -1 for "complementary"
    M1, M2 = K*(K-1)/2, K-1
    pc_loss = torch.sum(loss_matrix)*(K-1)/len(labels) - M1 + M2
    return pc_loss

def accuracy_check(loader, model):
    sm = F.softmax
    total, num_samples = 0, 0
    for images, labels in loader:
        labels, images = labels.to(device), images.to(device)
        outputs = model(images)
        sm_outputs = sm(outputs, dim=1)
        _, predicted = torch.max(sm_outputs.data, 1)
        total += (predicted == labels).sum().item()
        num_samples += labels.size(0)
    return 100 * total / num_samples

def chosen_loss_c(f, K, labels, ccp, meta_method):
    class_loss_torch = None
    if meta_method=='free':
        final_loss, class_loss_torch = assump_free_loss(f=f, K=K, labels=labels, ccp=ccp)
    elif meta_method=='nn':
        final_loss, class_loss_torch = non_negative_loss(f=f, K=K, labels=labels, beta=0, ccp=ccp)
    elif meta_method=='forward':
        final_loss = forward_loss(f=f, K=K, labels=labels)
    elif meta_method=='pc':
        final_loss = pc_loss(f=f, K=K, labels=labels)
    return final_loss, class_loss_torch
