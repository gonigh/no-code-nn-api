from string import Template
import os.path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
template = Template('''
import torch

import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import net
Net = net.Net


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = $lossTrain
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += $lossTest.item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    hyperParameters = $hyperParameters
    args = {
        'batch_size': hyperParameters["batchSize"],
        'epochs': hyperParameters["epochs"],
        'lr': hyperParameters["lr"],
        'gamma': hyperParameters["gamma"],
        'seed': hyperParameters["seed"],
        'no_cuda': hyperParameters["NoCUDA"],
        'test_batch_size': 1000,
        'log_interval': 10
    }
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args['batch_size']}
    test_kwargs = {'batch_size': args['test_batch_size']}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('./data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.$optimizerName(model.parameters(), lr=args['lr'])
    
    scheduler = StepLR(optimizer, step_size=1, gamma=args['gamma'])
    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()


if __name__ == '__main__':
    main()
''')

lossList = {
    'L1Loss': {'train': 'F.l1_loss(output, target)', 'test': 'F.l1_loss(output, target, reduction="sum")'},
    'CrossEntropy': {'train': 'F.cross_entropy(output, target)', 'test': 'F.cross_entropy(output, target, reduction="sum")'},
    'SmoothL1Loss': {'train': 'F.smooth_l1_loss(output, target)', 'test': 'F.smooth_l1_loss(output, target, reduction="sum")'},
    'MSELoss': {'train': 'F.mse_loss(output, target)', 'test': 'F.mse_loss(output, target, reduction="sum")'},
    'BCELoss': {'train': 'F.binary_cross_entropy(output, target)', 'test': 'F.binary_cross_entropy(output, target, reduction="sum")'},
    'BCEWithLogitsLoss': {'train': 'F.binary_cross_entropy_with_logits(output, target)', 'test': 'F.binary_cross_entropy_with_logits(output, target, reduction="sum")'},
    'NLLLoss': {'train': 'F.nll_loss(output, target)', 'test': 'F.nll_loss(output, target, reduction="sum")'},
    'KLDivLoss': {'train': 'F.kl_div(output, target)', 'test': 'F.kl_div(output, target, reduction="sum")'},
    'MarginRankingLoss': {'train': 'F.margin_ranking_loss(output, target)', 'test': 'F.margin_ranking_loss(output, target, reduction="sum")'},
    'MultiMarginLoss': {'train': 'F.multi_margin_loss(output, target)', 'test': 'F.multi_margin_loss(output, target, reduction="sum")'},
    'MultiLabelMarginLoss': {'train': 'F.multilabel_margin_loss(output, target)', 'test': 'F.multilabel_margin_loss(output, target, reduction="sum")'},
    'SoftMarginLoss': {'train': 'F.soft_margin_loss(output, target)', 'test': 'F.soft_margin_loss(output, target, reduction="sum")'},
    'MultiLabelSoftMarginLoss': {'train': 'F.multilabel_soft_margin_loss(output, target)', 'test': 'F.multilabel_soft_margin_loss(output, target, reduction="sum")'},
    'CosineEmbeddingLoss': {'train': 'F.cosine_embedding_loss(output, target)', 'test': 'F.cosine_embedding_loss(output, target, reduction="sum")'},
}

def createMain(loss, optim, hyper):
    file_path = ROOT_DIR + '\\output\\main.py'
    print(loss, optim, hyper)
    # 生成可执行的Python文件
    with open(file_path, 'w') as f:
        f.write(template.substitute(lossTrain=lossList[loss]['train'], lossTest=lossList[loss]['test'], optimizerName=optim, hyperParameters=hyper))
    with open(file_path, 'r') as f:
        s = f.read()
    print(s)
    return s