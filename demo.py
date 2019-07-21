import argparse, time, os
from utils_data import *
from utils_algo import *
from models import *

np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(
	prog='complementary-label learning demo file.',
	usage='Demo with complementary labels.',
	description='A simple demo file with MNIST dataset.',
	epilog='end',
	add_help=True)

parser.add_argument('-lr', '--learning_rate', help='optimizer\'s learning rate', default=5e-5, type=float)
parser.add_argument('-bs', '--batch_size', help='batch_size of ordinary labels.', default=256, type=int)
parser.add_argument('-me', '--method', help='method type. ga: gradient ascent. nn: non-negative. free: Theorem 1. pc: Ishida2017. forward: Yu2018.', choices=['ga', 'nn', 'free', 'pc', 'forward'], type=str, required=True)
parser.add_argument('-mo', '--model', help='model name', choices=['linear', 'mlp'], type=str, required=True)
parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=300)
parser.add_argument('-wd', '--weight_decay', help='weight decay', default=1e-4, type=float)
parser.add_argument('-p', '--parallel_gpus', help='Enable usage of multiple GPUs', action='store_true')

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

full_train_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, K = prepare_mnist_data(batch_size=args.batch_size)
ordinary_train_loader, complementary_train_loader, ccp = prepare_train_loaders(full_train_loader=full_train_loader, batch_size=args.batch_size, ordinary_train_dataset=ordinary_train_dataset)

meta_method = 'free' if args.method =='ga' else args.method

if args.model == 'mlp':
    model = mlp_model(input_dim=28*28, hidden_dim=500, output_dim=K)
elif args.model == 'linear':
    model = linear_model(input_dim=28*28, output_dim=K)

if torch.cuda.device_count() > 1 and args.parallel_gpus: model = nn.DataParallel(model)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr = args.learning_rate)

train_accuracy = accuracy_check(loader=train_loader, model=model)
test_accuracy = accuracy_check(loader=test_loader, model=model)
print('Epoch: 0. Tr Acc: {}. Te Acc: {}'.format(train_accuracy, test_accuracy))
save_table = np.zeros(shape=(args.epochs, 3))

for epoch in range(args.epochs):
    for i, (images, labels) in enumerate(complementary_train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss, loss_vector = chosen_loss_c(f=outputs, K=K, labels=labels, ccp=ccp, meta_method=meta_method)
        if args.method == 'ga':
            if torch.min(loss_vector).item() < 0:
                loss_vector_with_zeros = torch.cat((loss_vector.view(-1,1), torch.zeros(K, requires_grad=True).view(-1,1).to(device)), 1)
                min_loss_vector, _ = torch.min(loss_vector_with_zeros, dim=1)
                loss = torch.sum(min_loss_vector)
                loss.backward()
                for group in optimizer.param_groups:
                    for p in group['params']:
                        p.grad = -1*p.grad
            else:
                loss.backward()
        else:
            loss.backward()
        optimizer.step()
    train_accuracy = accuracy_check(loader=train_loader, model=model)
    test_accuracy = accuracy_check(loader=test_loader, model=model)
    print('Epoch: {}. Tr Acc: {}. Te Acc: {}.'.format(epoch+1, train_accuracy, test_accuracy))
    save_table[epoch, :] = epoch+1, train_accuracy, test_accuracy

np.savetxt(args.method+'_demo_results.txt', save_table, delimiter=',', fmt='%1.3f')
