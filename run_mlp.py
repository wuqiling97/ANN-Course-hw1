from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d


train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model definition here
# You should explore different model architecture
layer1_out = 128
model = Network()
model.add(Linear('fc1', 784, layer1_out, 0.01))
model.add(Relu('active1'))
model.add(Linear('fc2', layer1_out, 10, 0.01))


loss = EuclideanLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.01,
    'weight_decay': 0.0,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': 100,
    'disp_freq': 600,  # display info each disp_freq training
    'test_epoch': 5
}

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(config['max_epoch']+1):
    LOG_INFO('Training @ %d epoch...' % epoch)
    train_res = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
    train_loss.append(train_res[0])
    train_acc.append(train_res[1])

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % epoch)
        test_res = test_net(model, loss, test_data, test_label, config['batch_size'])
        test_loss.append(test_res[0])
        test_acc.append(test_res[1])
