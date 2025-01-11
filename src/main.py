import torch
import torchvision
import onnx
import visdom

import relu
import gelu
import tanh

data_train = torchvision.datasets.MNIST('../data/mnist',
                                        download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()]))
data_test = torchvision.datasets.MNIST('../data/mnist',
                                       train=False,
                                       download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.Resize((32, 32)),
                                           torchvision.transforms.ToTensor()]))
data_train_loader = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle=True, num_workers=8)
data_test_loader = torch.utils.data.DataLoader(data_test, batch_size=128, num_workers=8)

net_relu = relu.LeNet5()
net_gelu = gelu.LeNet5()
net_tanh = tanh.LeNet5()
optimizer = torch.optim.Adam(net_relu.parameters(), lr=2e-3)
optimizer_1 = torch.optim.Adam(net_gelu.parameters(), lr=2e-3)
optimizer_2 = torch.optim.Adam(net_tanh.parameters(), lr=2e-3)

vis = visdom.Visdom(port=8098)
loss_window = vis.line(Y=torch.zeros((1, 2)), opts=dict(title='Train Loss', xlabel='Epoch', ylabel='Loss', legend=['Model 1', 'Model 2']))
accuracy_window = vis.line(Y=torch.zeros((1, 2)), opts=dict(title='Test Accuracy', xlabel='Epoch', ylabel='Accuracy', legend=['Model 1', 'Model 2']))

def train(epoch, model, optimizer, max_batches=30):
    model.train()
    loss_list = []
    for i, (images, labels) in enumerate(data_train_loader):
        if i >= max_batches:
            break
        optimizer.zero_grad()
        output = model(images)
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss_list.append(loss.detach().cpu().item())
        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))
        loss.backward()
        optimizer.step()
    avg_loss = sum(loss_list) / len(loss_list) if loss_list else 0
    return avg_loss

def test(model):
    model.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for images, labels in data_test_loader:
            output = model(images)
            avg_loss += torch.nn.functional.cross_entropy(output, labels, reduction='sum').item()
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
    avg_loss /= len(data_test)
    accuracy = float(total_correct) / len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss, accuracy))
    return accuracy

def train_and_test(epoch):
    avg_loss1 = train(epoch, net_relu, optimizer)
    accuracy1 = test(net_relu)
    avg_loss2 = train(epoch, net_gelu, optimizer_1)
    accuracy2 = test(net_gelu)
    avg_loss3 = train(epoch, net_tanh, optimizer_1)
    accuracy3 = test(net_tanh)
    vis.line(X=torch.ones((1, 2)) * epoch, Y=torch.Tensor([[avg_loss1, avg_loss2, avg_loss3]]), win=loss_window, update='append')
    vis.line(X=torch.ones((1, 2)) * epoch, Y=torch.Tensor([[accuracy1, accuracy2, accuracy3]]), win=accuracy_window, update='append')
    dummy_input = torch.randn(1, 1, 32, 32, requires_grad=True)
    torch.onnx.export(net_relu, dummy_input, "tju_nn_lenet_base.onnx")
    onnx_model = onnx.load("tju_nn_lenet_base.onnx")
    onnx.checker.check_model(onnx_model)
    torch.onnx.export(net_gelu, dummy_input, "tju_nn_lenet_base_model1.onnx")
    onnx_model_1 = onnx.load("tju_nn_lenet_base_model1.onnx")
    onnx.checker.check_model(onnx_model_1)

def main():
    for epo in range(1, 6):
        train_and_test(epo)

if __name__ == '__main__':
    main()
