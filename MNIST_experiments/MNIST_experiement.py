from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt

from attacks import fgsm_attack

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

BATCH_SIZE = 100
USE_CUDA = True
NUM_EPOCHS = 10

print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
dtype = torch.float32 
adv_train = True

# Load the train and test set
train_dataset = datasets.MNIST(root='data',train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def adv_examples_gen(model, data, target, epsilon):
    adv_count = 0
    if adv_train:
        scores = model(data)
        loss = F.cross_entropy(scores, target)
        optimizer.zero_grad()
        loss.backward()
        if random.uniform(0, 1) <= 0.5:
            adv_count += 1
            data_grad = data.grad.data
            # Call FGSM Attack, using epsilon 0.2
            data = fgsm_attack(data, 0.2, data_grad)
    return adv_count, data

# Normal training
def training(model, optimizer, epochs=NUM_EPOCHS):
    adv_count = 0
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data.requires_grad = True
            
            adv_count, data = adv_examples_gen(model, data, target, 0.2)
            ### FORWARD AND BACK PROP
            scores = model(data)
            loss = F.cross_entropy(scores, target)
            optimizer.zero_grad()
            loss.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()
            
            ### LOGGING
            if not batch_idx % 50:
                print ('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f' %(epoch+1, NUM_EPOCHS, batch_idx, len(train_loader), loss))

        model.eval()
        with torch.set_grad_enabled(False): # save memory during inference
            print('Epoch: %03d/%03d | Train: %.3f%%' % (epoch+1, NUM_EPOCHS, compute_accuracy(model, train_loader, device=device)))
    # Compute accuracy for test set
    with torch.set_grad_enabled(False): # save memory during inference
        print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader, device=device)))
        print('Adv_count: %d' % adv_count)

def compute_accuracy(model, loader, device):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for data, target in loader:
            #data, target = data.to(device, dtype=dtype), target.to(device, dtype=torch.long)
            data, target = data.to(device), target.to(device)
            scores = model(data)
            _, preds = scores.max(1)
            num_correct += (preds == target).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
    return acc * 100

# Initialize the model
model = models.resnet50(pretrained=False, progress=True).to(device)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.load_state_dict(torch.load("/homes/yx3017/Desktop/Individual_project/Individual_project/MNIST_experiments/model.pt", map_location='cpu'))

optimizer = optim.Adam(model.parameters())
# training(model, optimizer, epochs = 10)
#torch.save(model.state_dict(), 'adv_model.pt')

# Pertubate the test set with adversarial attacks and check accuracy
def adv_test(model, device, test_loader, epsilon):
    model = model.to(device)
    correct = 0
    adv_examples = []
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        _, init_pred = output.max(1)

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data
        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        for i in range(len(target)):
            if final_pred[i].item() == target[i].item():
                correct += 1
                # Special case for saving 0 epsilon examples
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data[i].squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred[i].item(), final_pred[i].item(), adv_ex) )
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data[i].squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred[i].item(), final_pred[i].item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/10000
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, 10000, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


accuracies = []
examples = []
epsilons = [0, .01, .05, .1, .15, .2, .25, .3]

# Run test for each epsilon
for eps in epsilons:
    acc, ex = adv_test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

# # Plot the figure of accuracy against epsilon
# plt.figure(figsize=(5,5))
# plt.plot(epsilons, accuracies, "*-")
# plt.yticks(np.arange(0, 1.1, step=0.1))
# plt.xticks(np.arange(0, .6, step=0.05))
# plt.title("Accuracy vs Epsilon")
# plt.xlabel("Epsilon")
# plt.ylabel("Accuracy")
# plt.show()

# Show examples of adversarial examples and prediction change
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("{}".format(epsilons[i]), fontsize=12)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.savefig('/homes/yx3017/Desktop/Individual_project/Individual_project/MNIST_experiments/adv_number_example.png')