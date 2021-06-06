import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import models, datasets, transforms

from monai.config import print_config
from monai.transforms import \
    Compose, LoadPNG, AddChannel, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom
from monai.networks.nets import densenet121
from monai.metrics import compute_roc_auc

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
#torch.use_deterministic_algorithms(True)

batch_size = 199
USE_CUDA = True
epoch_num = 5
val_interval = 1
adv_train = True

# Loading images
data_dir = "/vol/bitbucket/yx3017/MedNIST"
class_names = sorted([x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))])
num_class = len(class_names)
image_files = [[os.path.join(data_dir, class_name, x) 
                for x in os.listdir(os.path.join(data_dir, class_name))] 
               for class_name in class_names]
image_file_list = []
image_label_list = []
for i, class_name in enumerate(class_names):
    image_file_list.extend(image_files[i])
    image_label_list.extend([i] * len(image_files[i]))
num_total = len(image_label_list)
image_width, image_height = Image.open(image_file_list[0]).size

print('Total image count:', num_total)
print("Image dimensions:", image_width, "x", image_height)
print("Label names:", class_names)
print("Label counts:", [len(image_files[i]) for i in range(num_class)])

# # Visualization
# plt.subplots(3, 3, figsize=(8, 8))
# for i,k in enumerate(np.random.randint(num_total, size=9)):
#     im = Image.open(image_file_list[k])
#     arr = np.array(im)
#     plt.subplot(3, 3, i + 1)
#     plt.xlabel(class_names[image_label_list[k]])
#     plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
# plt.tight_layout()
# plt.show()

# Split training/validation and test set
valid_frac, test_frac = 0.1, 0.1
trainX, trainY = [], []
valX, valY = [], []
testX, testY = [], []

for i in range(num_total):
    rann = np.random.random()
    if rann < valid_frac:
        valX.append(image_file_list[i])
        valY.append(image_label_list[i])
    elif rann < test_frac + valid_frac:
        testX.append(image_file_list[i])
        testY.append(image_label_list[i])
    else:
        trainX.append(image_file_list[i])
        trainY.append(image_label_list[i])

print("Training count =",len(trainX),"Validation count =", len(valX), "Test count =",len(testX))

# Transformations
train_transforms = Compose([
    LoadPNG(image_only=True),
    AddChannel(), # Adds a 1-length channel dimension to the input image so the image can be correctly interpreted by other transforms.
    ScaleIntensity(),
    RandRotate(range_x=15, prob=0.5, keep_size=True),
    RandFlip(spatial_axis=0, prob=0.5),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
    ToTensor()
])

val_transforms = Compose([
    LoadPNG(image_only=True),
    AddChannel(),
    ScaleIntensity(),
    ToTensor()
])

# Creating datasets and dataloaders
class MedNISTDataset(Dataset):

    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

train_ds = MedNISTDataset(trainX, trainY, train_transforms)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)

val_ds = MedNISTDataset(valX, valY, val_transforms)
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=8)

test_ds = MedNISTDataset(testX, testY, val_transforms)
test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=8)


print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
#model = models.resnet50(pretrained=False, progress=True).to(device)
#model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)
model = densenet121(spatial_dims=2,in_channels=1, out_channels=num_class).to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)

def adv_backwards(model, data, target, optimizer):
    scores = model(data)
    loss = F.cross_entropy(scores, target)
    optimizer.zero_grad()
    loss.backward()

# FGSM attack
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Return the perturbed image
    return perturbed_image


def bim_attack(image, target, epsilon, data_grad, num_steps):
    # Set step size as the middle point of epsilon/num_steps and epsilon
    step_size = 0.5 * (epsilon/num_steps + epsilon)
    for i in range(num_steps):
        perturbed_image = fgsm_attack(image, step_size, data_grad).clone().detach().requires_grad_(True)
        adv_backwards(model, perturbed_image, target, optimizer)
        data_grad = perturbed_image.grad.data
    return perturbed_image

def pgd_attack(image, target, epsilon, data_grad, num_steps):
    step_size = 0.5 * (epsilon/num_steps + epsilon)

    batch, ch, row, col = image.shape
    uni_noise = np.random.uniform(-epsilon, epsilon, (batch, row, col))
    uni_noise = torch.from_numpy(uni_noise.reshape(batch, ch, row, col)).to(device, dtype=torch.float)
    
    image = image + uni_noise
    for i in range(num_steps):
        
        perturbed_image = fgsm_attack(image, step_size, data_grad).clone().detach().requires_grad_(True)
        clipped_delta = torch.clamp(perturbed_image.data - image.data, -epsilon, epsilon) #clipping the delta
        perturbed_image = torch.tensor(image.data + clipped_delta).clone().detach().requires_grad_(True)
        adv_backwards(model, perturbed_image, target, optimizer)
        data_grad = perturbed_image.grad.data
    
    return perturbed_image

def cw_l2_attack(model, images, target, c=1e-4, kappa=0, lr=0.01):
    images = images.to(device)
    target = target.to(device)

    def f(x):
        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())

        return torch.clamp(j-i, min=-kappa)
    
    w = torch.zeros_like(images, requires_grad=True).to(device)
    optimizer = optim.Adam([w], lr=lr)
    prev = 1e10
    
    for step in range(max_iter) :
        a = 1/2*(nn.Tanh()(w) + 1)
        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost
        
        print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')

    attack_images = 1/2*(nn.Tanh()(w) + 1)

    return attack_images

# Generating adversarial examples
def adv_examples_gen(model, data, target, epsilon, attack_name):
    adv_backwards(model, data, target, optimizer)
    
    data_grad = data.grad.data
    # Call Attacks, using epsilon specified
    if attack_name == "fgsm":
        data = fgsm_attack(data, epsilon, data_grad)
    elif attack_name == 'bim':
        data = bim_attack(data, target, epsilon, data_grad, 3)
    elif attack_name == 'pgd':
        data = pgd_attack(data, target, epsilon, data_grad, 3)
    # # elif attack_name == 'cw':
    #     data = cw_l2_attack(model, data, target)
    else:
        print('Wrong attack name input')
        data = None
    return data

def train(epoch_num, model, train_loader, val_loader, name, percentage, attack_name, epsilon):
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    

    for epoch in range(epoch_num):
        adv_count = 0
        model = model.to(device)
        print('-' * 10)
        print("epoch: {} / {}".format(epoch + 1, epoch_num))
        model.train()
        epoch_loss = 0
        step = 0
        for i, (inputs, labels) in enumerate(train_loader):
            step += 1
            inputs, labels = inputs.to(device), labels.to(device)
            inputs.requires_grad = True
            optimizer.zero_grad()
            if adv_train:
                if i <= percentage * len(train_loader):
                    inputs = adv_examples_gen(model, inputs, labels, epsilon, attack_name)
                    adv_count = i

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")

            epoch_len = len(train_ds) // train_loader.batch_size
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        print("Adversarial count = ", str(adv_count))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                auc_metric = compute_roc_auc(y_pred, y, to_onehot_y=True, softmax=True)
                metric_values.append(auc_metric)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)

                if auc_metric > best_metric:
                    best_metric = auc_metric
                    best_metric_epoch = epoch + 1
                    with torch.no_grad():
                        torch.save(model.state_dict(), '/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/'+name+'.pth')
                    print('saved new best metric model')
                print(f"current epoch: {epoch + 1} current AUC: {auc_metric:.4f}"
                    f" current accuracy: {acc_metric:.4f} best AUC: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}")
        
                    
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    

#train(epoch_num, model, train_loader, val_loader, name, 0.2)

# model.load_state_dict(torch.load('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/MedNIST_model.pth'))
# model.eval()

def normal_testing(model, device, test_loader):
    y_true = list()
    y_pred = list()
    acc = [0 for c in class_names]
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())
    
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm.diagonal())


# Pertubate the test set with adversarial attacks and check accuracy
def adv_test(model, device, test_loader, epsilon, attack_name, percentage):
    model = model.to(device)
    correct = 0
    adv_examples = []
    y_true  = list()
    y_pred = list()

    acc = {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0}
    class_num = [1039, 910, 987, 1033, 967, 1050]

    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        _, init_pred = output.max(1)

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data
        if i <= percentage * len(test_loader):
            # Call Attacks
            if attack_name == "fgsm":
                perturbed_data = fgsm_attack(data, epsilon, data_grad)
            elif attack_name == 'bim':
                 perturbed_data = bim_attack(data, target, epsilon, data_grad, 3)
            elif attack_name == 'pgd':
                perturbed_data = pgd_attack(data, target, epsilon, data_grad, 3)
            # elif attack_name == 'cw':
            #     data = cw_l2_attack(model, data, target)
            else:
                print('Wrong attack name input')
                perturbed_data = None
        else:
            perturbed_data = data
        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        for i in range(len(target)):
            if final_pred[i].item() == target[i].item():
                correct += 1
                acc[final_pred[i].item()] += 1
            else:
                if i == 0:
                    adv_ex = perturbed_data[i].squeeze().detach().cpu().numpy()
                    adv_examples.append( adv_ex )
            y_true.append(target[i].item())
            y_pred.append(final_pred[i].item())

    # Calculate final accuracy for this epsilon
    final_acc = correct/len(testX)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(testX), final_acc))

    print("=============")

    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # print(confusion_matrix(y_true, y_pred))
    
    # for i in range(6):
    #     acc[i] = acc[i] / class_num[i]
    # print(acc)

    # print("=============")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


########## Experiment 1: Train normally, test on pertubated, does it affect the accuracy? ############
# Show examples of image:

# plt.subplots(2, 3, figsize=(8, 8))
# for i,k in enumerate(np.random.randint(num_total, size=9)):
#     im = Image.open(image_file_list[k])
#     arr = np.array(im)
#     plt.subplot(2, 3, i + 1)
#     plt.xlabel(class_names[image_label_list[k]])
#     plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
# plt.tight_layout()
# plt.savefig('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_1/normal_example.png')
# adv_train = False
# #train(epoch_num, model, train_loader, val_loader, 'Normally_trained')
# model.load_state_dict(torch.load('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_1/Normally_trained.pth'))
# model.eval()

# adv_test(model, device, test_loader, 0.1, 'pgd', 1)

# normal_testing(model, device, test_loader)
# adv_test(model, device, test_loader, 0.1, 'fgsm', 1)

# normal = [99, 100, 99, 100, 99, 99]
# adv = [7.8, 0, 17.8, 98.1, 84.6, 4.8]

# x = np.arange(len(class_names))  # the label locations
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, normal, width, label='Normal test')
# rects2 = ax.bar(x + width/2, adv, width, label='Adversarial test')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Accuracy of model(%)')
# ax.set_title('Accuracy comparison per class')
# ax.set_xticks(x)
# ax.set_xticklabels(class_names)
# ax.legend()

# # ax.bar_label(rects1, padding=3)
# # ax.bar_label(rects2, padding=3)

# fig.tight_layout()

# plt.savefig('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_1/performance_comparison_bar.png')
# normal = [99.78, 99.78, 99.78, 99.78]
# adv = [50.2, 35.73, 25.04, 35.73]

# labels = ['Precision', 'Recall', 'F1-score', 'Accuracy']

# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/4, normal, width, label='Normal test')
# rects2 = ax.bar(x + width/4, adv, width, label='Adversarial test')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Percentage(%)')
# ax.set_title('Overall performance comparison')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()

# # ax.bar_label(rects1, padding=3)
# # ax.bar_label(rects2, padding=3)

# fig.tight_layout()

# plt.savefig('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_1/overall_comparison_bar.png')

# # How does epsilon affects the accuracy for one attack?
# accuracies_fgsm, accuracies_pgd, accuracies_cw = [], [], []
epsilons = [0, .01, .05, .1, .15, .2, .25, .3]
# # #  
# examples = [[] for i in range(len(epsilons))]
# accuracies_fgsm, accuracies_bim, accuracies_pgd = [], [], []
# for i in range(len(epsilons)):
#     acc, ex = adv_test(model, device, test_loader, epsilons[i], 'fgsm', 1)
#     accuracies_fgsm.append(acc)
#     #examples[i].append(ex[0])

#     acc, ex = adv_test(model, device, test_loader, epsilons[i], 'bim', 1)
#     accuracies_bim.append(acc)
#     #examples[i].append(ex[0])

#     acc, ex = adv_test(model, device, test_loader, epsilons[i], 'pgd', 1)
#     accuracies_pgd.append(acc)
#     #examples[i].append(ex[0])

#     print("================================================")

# accuracies_fgsm = [0.998329435349148, 0.9485466087537587, 0.644503842298697, 0.36618777146675574, 0.29836284664216506, 0.18392916805880388, 0.1506849315068493, 0.1506849315068493]
# accuracies_bim = [0.998329435349148, 0.958904109589041, 0.5962245238890745, 0.3504844637487471, 0.2216839291680588, 0.11693952555963916, 0.08002004677581022, 0.0982292014700969]
# accuracies_pgd = [0.998329435349148, 0.9565653190778484, 0.48128967591045774, 0.24757768125626461, 0.07801536919478784, 0.06064149682592716, 0.09188105579685933, 0.12245238890745072]

# plt.figure(figsize=(5,5))
# plt.plot(epsilons, accuracies_fgsm, "*-", label='FGSM')
# plt.plot(epsilons, accuracies_bim, "*-", label='BIM')
# plt.plot(epsilons, accuracies_pgd, "*-", label='PGD')
# plt.legend()
# plt.yticks(np.arange(0, 1.1, step=0.1))
# plt.xticks(np.arange(0, .35, step=0.05))
# plt.title("How do epsilons affect the accuracy of the model?")
# plt.xlabel("Epsilon")
# plt.ylabel("Accuracy on test set")
# plt.savefig('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_1/Accuracy_vs_Epsilon_new.png')

# ==============================================

# Examples of each attack
# cnt = 0
# fig = plt.figure(figsize=(12, 6))
# fig.suptitle("Examples of adversarial images", fontsize=16)
# sample = next(iter(test_loader))
# data, target = sample[0].to(device), sample[1].to(device)

# data.requires_grad = True
# for j in range(3):
#     for i in range(len(epsilons)):
#         output = model(data)
#         _, init_pred = output.max(1)

#         loss = F.nll_loss(output, target)
#         model.zero_grad()
#         loss.backward()

#         data_grad = data.grad.data
#         adv_ex = fgsm_attack(data, epsilons[i], data_grad).squeeze().detach().cpu().numpy()
#         examples[j].append(adv_ex[0])
#         adv_ex = bim_attack(data, target, epsilons[i], data_grad, 3).squeeze().detach().cpu().numpy()
#         examples[j].append(adv_ex[0])
#         adv_ex = pgd_attack(data, target, epsilons[i], data_grad, 3).squeeze().detach().cpu().numpy()
#         examples[j].append(adv_ex[0])

#         cnt += 1
#         ax = plt.subplot(3, len(epsilons),cnt)
#         plt.xticks([], [])
#         plt.yticks([], [])

#         if j == 0:
#             ax.title.set_text(epsilons[i])
#         if i == 0 and j == 0:
#             plt.ylabel('FGSM')
#         if i == 0 and j == 1:  
#             plt.ylabel('BIM')
#         if i == 0 and j == 2:  
#             plt.ylabel('PGD')
        
#         ex = examples[j][i]

#         plt.imshow(ex, cmap="gray", aspect='auto')


# plt.tight_layout()
# plt.savefig('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_1/adverexample_new_test.png')


# Close up comparison between two images
# cnt = 0
# fig = plt.figure(figsize=(6, 3))
# fig.suptitle("Comparison between clean and adversarial images", fontsize=16)
# sample = next(iter(test_loader))
# data, target = sample[0].to(device), sample[1].to(device)

# data.requires_grad = True
# output = model(data)
# _, init_pred = output.max(1)

# loss = F.nll_loss(output, target)
# model.zero_grad()
# loss.backward()

# data_grad = data.grad.data
# adv_ex = fgsm_attack(data, 0.1, data_grad).squeeze().detach().cpu().numpy()

# plt.subplot(1, 2, 1)
# plt.imshow(data[0].squeeze().detach().cpu().numpy(), cmap="gray")

# plt.subplot(1, 2, 2)
# plt.imshow(adv_ex[0], cmap="gray")

# # plt.tight_layout(pad=1.0)
# plt.savefig('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_1/epsilons/close_up.png')
# =======================================================

########## Experiment 2.1: adding fsgm attacked images into the training set, improve accuracy against adv attacked test set? ############
# Compare the accuracy of fgsm-attack-trained model and pgd-attack-trained model
# adv_train = True
# accuracies = []
# epsilons = [0, .05, .1, .15, .2, .25, .3]

# # train(epoch_num, model, train_loader, val_loader, 'experiment_2/Adversarial_training_help/fgsm_trained', 1, 'fgsm', 0.1)

# model.load_state_dict(torch.load('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_2/Adversarial_training_help/fgsm_trained.pth'))
# model.eval()

# print("======== FGSM trained, test on FGSM test set ===========")
# acc, _ = adv_test(model, device, test_loader, 0.1, 'fgsm', 1)
# print("Adversarial training using FGSM testing on FGSM set achieves an accuracy of: " + str(acc) + " on the FGSM test set.")
# print("========================================================")
# print()
# print()

# model.load_state_dict(torch.load('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_2/Adversarial_training_help/fgsm_trained.pth'))
# model.eval()

# print("======== FGSM trained, test on PGD test set ===========")
# acc, _ = adv_test(model, device, test_loader, 0.1, 'pgd', 1)
# print("Adversarial training using FGSM testing on PGD set achieves an accuracy of: " + str(acc) + " on the PGD test set.")
# print("========================================================")
# print()
# print()


# train(epoch_num, model, train_loader, val_loader, 'experiment_2/Adversarial_training_help/pgd_trained', 1, 'pgd', 0.1)

# # model.load_state_dict(torch.load('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_2/Adversarial_training_help/pgd_trained.pth'))
# model.eval()
# print("======== PGD trained, test on FGSM test set ===========")
# acc, _ = adv_test(model, device, test_loader, 0.1, 'fgsm', 1)
# print("Adversarial training using PGD testing on FGSM set achieves an accuracy of: " + str(acc) + " .")
# print("========================================================")
# print()
# print()




# # model.load_state_dict(torch.load('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_2/Adversarial_training_help/pgd_trained.pth'))
# model.eval()
# print("======== PGD trained, test on PGD test set ===========")
# acc, _ = adv_test(model, device, test_loader, 0.1, 'pgd', 1)
# print("Adversarial training using PGD testing on PGD achieves an accuracy of: " + str(acc) + " .")
# print("========================================================")
# print()
# print()

# FGSM_FGSM = [94.58, 94.05, 94.09, 94.05] 
# FGSM_PGD = [83.83, 79.30, 78.92, 79.30]
# PGD_PGD = [90.84, 90.46, 90.50, 90.46]
# PGD_FGSM = [88.65, 86.33, 85.52, 86.33]

# labels = ['Precision', 'Recall', 'F1-score', 'Accuracy']

# x = np.arange(len(labels))  # the label locations
# width = 0.2  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - 0.3, FGSM_FGSM, width, label='FGSM+FGSM', color='midnightblue')
# rects2 = ax.bar(x - 0.1, FGSM_PGD, width, label='FGSM+PGD', color='royalblue')
# rects3 = ax.bar(x + 0.1, PGD_PGD, width, label='PGD+PGD', color='cornflowerblue')
# rects4 = ax.bar(x + 0.3, PGD_FGSM, width, label='PGD+FGSM', color='lightsteelblue')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Percentage(%)')
# ax.set_title('Overall transferabilty performance')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()

# # ax.bar_label(rects1, padding=3)
# # ax.bar_label(rects2, padding=3)

# fig.tight_layout()


# plt.savefig('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_2/Adversarial_training_help/transfer_bar.png')

# ======================================= Experiment 1.2: train on adversarial test on clean =================
# adv_train = True
# train(epoch_num, model, train_loader, val_loader, 'experiment_1/all_adversarial/pgd_0.2', 1, 'pgd', 0.2)
# normal_testing(model, device, test_loader)


# normal = [50.22, 35.73, 25.04, 35.73]
# adv = [94.58, 94.05, 94.09, 94.05]
# labels = ['Precision','Recall', 'F1-score', 'Accuracy']
# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, normal, width, label='Standard')
# rects2 = ax.bar(x + width/2, adv, width, label='FGSM-trained')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Performance of the models(%)')
# ax.set_title('Standard/FGSM-trained model against same test set')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()

# # ax.bar_label(rects1, padding=3)
# # ax.bar_label(rects2, padding=3)

# fig.tight_layout()

# plt.savefig('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_2/Adversarial_training_help/fgsm_comparison.png')

# model.load_state_dict(torch.load('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_1/Normally_trained.pth'))
# model.eval()
# adv_test(model, device, test_loader, 0.1, 'pgd', 1)

# normal = [26.12, 25.01, 14.26, 25.01]
# adv = [90.84, 90.46, 90.5, 90.46]
# labels = ['Precision','Recall', 'F1-score', 'Accuracy']
# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, normal, width, label='Standard')
# rects2 = ax.bar(x + width/2, adv, width, label='PGD-trained')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Performance of the models(%)')
# ax.set_title('Standard/PGD-trained model against same test set')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()

# # ax.bar_label(rects1, padding=3)
# # ax.bar_label(rects2, padding=3)

# fig.tight_layout()

# plt.savefig('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_2/Adversarial_training_help/pgd_comparison.png')



# ======================================= Experiment 2.2: compare epsilons ======================
# accuracies = []
epsilons = [i/100 for i in range(0, 35, 5)]
# # Compare different epsilons
# for i in range(0, 35, 5):
#     # train(epoch_num, model, train_loader, val_loader, 'experiment_2/epsilons/pgd+fgsm/pgd+fgsm_epsilon' + str(i/100), 1, 'pgd', i/100)
#     model.load_state_dict(torch.load('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_2/epsilons/fgsm+fgsm/fgsm_epsilon' + str(i/100) + '.pth'))
#     model.eval()
#     acc_list = []
#     for j in range(0, 35, 5):
#         acc, _ = adv_test(model, device, test_loader, j/100, 'fgsm', 1)
#         acc_list.append(round(acc, 4))
    
#     accuracies.append(acc_list)
# print(accuracies)


# mean1, mean2, mean3 = [], [], []
# # FGSM+FGSM:
fgsm_fgsm = [[0.9987, 0.6756, 0.41, 0.2471, 0.2235, 0.2265, 0.2163], [0.9968, 0.9621, 0.7698, 0.5122, 0.3799, 0.219, 0.1923], 
[0.9779, 0.9818, 0.9215, 0.7001, 0.5124, 0.3662, 0.2653], [0.8496, 0.8082, 0.9838, 0.9562, 0.8116, 0.6345, 0.4511], 
[0.7399, 0.5578, 0.8249, 0.9686, 0.94, 0.8182, 0.6811], [0.6255, 0.5277, 0.5862, 0.9178, 0.856, 0.9061, 0.7822], [0.5494, 0.3899, 0.5175, 0.6385, 0.8705, 0.8059, 0.8076]]

fgsm_fgsm_t = np.array(fgsm_fgsm).transpose()
# best = [[0.0, 0.10, 0.15, 0.20, 0.20, 0.25, 0.3], [0.0, 0.1, 0.15, 0.25, 0.3, 0.3, 0.3], [0.0,0.05, 0.15, 0.3, 0.25, 0.25, 0.25], [0.0, 0.15, 0.20,0.20, 0.20, 0.20, 0.20]]
# labels = ['FGSM+FGSM', 'FGSM+PGD', 'PGD+PGD', 'PGD+FGSM']

# plt.figure(figsize=(5,5))
# for i in range(4):
#     plt.plot(epsilons, best[i], "*-", label=labels[i])

# plt.legend()
# plt.yticks(np.arange(0, 0.35, step=0.05))
# plt.xticks(np.arange(0, .35, step=0.05))
# plt.title("Relationships between training and testing epsilons")
# plt.xlabel("Attack epsilon")
# plt.ylabel("Best training epsilon")
# plt.savefig('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_2/epsilons/fgsm+fgsm/ep_comparison.png')




# # FGSM+PGD:
# fgsm_pgd = [[0.9982, 0.5576, 0.2392, 0.1712, 0.1457, 0.0381, 0.005], [0.9967, 0.9475, 0.6263, 0.4355, 0.2633, 0.1804, 0.1727], 
# [0.9442, 0.9703, 0.8953, 0.5379, 0.4462, 0.3009, 0.2035], [0.8281, 0.7875, 0.9013, 0.5755, 0.4569, 0.3936, 0.2581], 
# [0.7265, 0.511, 0.7997, 0.7145, 0.4069, 0.2771, 0.2718], [0.6091, 0.4056, 0.4591, 0.8259, 0.6415, 0.4332, 0.292], [0.6057, 0.4106, 0.3872, 0.4855, 0.7857, 0.5636, 0.3388]]

# fgsm_pgd_t = np.array(fgsm_pgd).transpose()

# # PGD+PGD:
# pgd_pgd = [[0.9985, 0.5521, 0.2259, 0.1702, 0.0762, 0.0057, 0.0286], [0.9926, 0.9452, 0.6888, 0.442, 0.2618, 0.1893, 0.1727], 
# [0.9774, 0.9359, 0.8867, 0.7305, 0.5461, 0.4387, 0.3329], [0.9773, 0.9437, 0.8951, 0.8131, 0.7095, 0.5416, 0.361], 
# [0.9556, 0.9145, 0.8724, 0.8228, 0.7604, 0.6811, 0.5212], [0.9208, 0.8901, 0.8598, 0.8336, 0.8034, 0.7594, 0.6808], [0.9262, 0.8963, 0.8717, 0.8393, 0.7952, 0.7427, 0.6679]]     

# pgd_pgd_t = np.array(pgd_pgd).transpose()

# # PGD+FGSM:
# pgd_fgsm = [[0.9945, 0.5518, 0.3343, 0.2355, 0.221, 0.2431, 0.2474], [0.993, 0.9424, 0.7406, 0.561, 0.3475, 0.2117, 0.1824], [0.9856, 0.9607, 0.92, 0.8532, 0.729, 0.6183, 0.5152], 
# [0.9853, 0.9644, 0.9405, 0.9021, 0.8403, 0.7147, 0.5605], [0.9818, 0.9574, 0.9526, 0.9415, 0.9238, 0.8801, 0.8004], 
# [0.5778, 0.5052, 0.4901, 0.7165, 0.7668, 0.7469, 0.6848], [0.5436, 0.4749, 0.4641, 0.4901, 0.6497, 0.6488, 0.5989]]
# pgd_fgsm_t = np.array(pgd_fgsm).transpose()




# for l in pgd_pgd:
#     mean2.append(np.average(l))
# for l in pgd_fgsm:
#     mean3.append(np.average(l))


# fig, ax = plt.subplots()
# rows, cols = [str(i/100) for i in range(0, 35, 5)], [str(i/100) for i in range(0, 35, 5)]

# rcolors = plt.cm.BuPu(np.full(len(rows), 0.1))
# ccolors = plt.cm.BuPu(np.full(len(cols), 0.1))
# table = ax.table(cellText=accuracies, rowLabels=rows, rowColours=rcolors, colLabels=cols, colColours=ccolors, loc='center')
# table[(1, 0)].set_facecolor("#56b5fd")
# table[(2, 0)].set_facecolor("#56b5fd")
# table[(3, 1)].set_facecolor("#56b5fd")
# table[(4, 2)].set_facecolor("#56b5fd")
# table[(5, 3)].set_facecolor("#56b5fd")
# table[(6, 3)].set_facecolor("#56b5fd")
# table[(7, 4)].set_facecolor("#56b5fd")

# ax.set_xlabel("Epsilon used in testing")
# ax.set_ylabel("Epsilon used in training")
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)

# table.scale(1, 1.5)
# plt.box(on=None)

# plt.title("Epsilons used in training and testing")
# plt.savefig('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_2/epsilons/epsilon_train_vs_test.png')


# plt.figure(figsize=(6,6))
# for i in range(7):
#     plt.plot(epsilons, fgsm_fgsm[i], "*-", label='eps: '+str(epsilons[i]))
# plt.legend()
# plt.yticks(np.arange(0, 1.1, step=0.1))
# plt.xticks(np.arange(0, .35, step=0.05))
# plt.title("Epsilon in adversarial training vs performance")
# plt.xlabel("Epsilon in the test set")
# plt.ylabel("Accuracy on test set")
# plt.savefig('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_2/epsilons/test_epsilon_vs_accuracy.png')
eps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
highlights = [[0.1, 2], [0.15, 3], [0.2, 4], [0.2, 4], [0.25, 5], [0.3, 6]]
fig, axs = plt.subplots(2, 3, figsize=(9,6))
for i in range(2):
    for j in range(3):
        xs = fgsm_fgsm_t[i * 3 + j +1]
        axs[i, j].plot(epsilons, xs)
        axs[i, j].scatter(highlights[i * 3 + j][0], xs[highlights[i * 3 + j][1]])
        axs[i, j].set_yticks(np.arange(0, 1.1, step=0.1))
        axs[i, j].set_xticks(np.arange(0, .35, step=0.05))
        axs[i, j].set_xticklabels(eps, rotation=45, ha="right",  rotation_mode="anchor")
        axs[i, j].set_title(str(epsilons[i * 3 + j +1]) + " ep attacked test set")
        axs[i, j].set(xlabel='Training epsilon', ylabel='Accuracy')


# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.savefig('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_2/epsilons/test_epsilon_vs_accuracy_separate.png')


################## Experiment 2.3: checking how much adversarial training helps ##################################

# # Use FGSM attack with epsilon 0.1 in the training phase, then test on 100% adversarially pertubated test set
# for percentage in percentages:
#     #train(epoch_num, model, train_loader, val_loader, 'Adversarially_trained_' + str(percentage), percentage)
#     model.load_state_dict(torch.load('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/Adversarially_trained_' + str(percentage) + '.pth'))
#     model.eval()
#     acc, _ = adv_test(model, device, test_loader, 0.1, 'fgsm', 0.5)
#     print("With percentage: " + str(percentage) + " of adversarial training it achieves accuracy of: " + str(acc) + " .")
#     accuracies.append(acc)

# accuracies = [0.5803541597059806, 0.7467423989308386, 0.7783160708319412, 0.9139659204811226, 0.9341797527564317, 0.9513865686602071, 0.9632475776812562, 0.9488807216839291, 0.975609756097561, 0.9667557634480455]
# plt.figure(figsize=(5,5))
# plt.plot(percentages, accuracies, "*-", label='FGSM')
# plt.legend()
# plt.yticks(np.arange(0.4, 1.1, step=0.1))
# plt.xticks(np.arange(0, 1.1, step=0.1))
# plt.title("Accuracy vs Percentage")
# plt.xlabel("Percentage of adversarial images in training set")
# plt.ylabel("Accuracy")
# plt.savefig('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/Results/Accuracy_vs_Percentage.png')

# # Use FGSM attack with epsilon 0.1 in the training phase, then test on 50% adversarially pertubated test set
# for percentage in percentages:
#     #train(epoch_num, model, train_loader, val_loader, 'Adversarially_trained_' + str(percentage), percentage)
#     model.load_state_dict(torch.load('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/Adversarially_trained_' + str(percentage) + '.pth'))
#     model.eval()
#     acc, _ = adv_test(model, device, test_loader, 0.1, 'fgsm', 0.1)
#     print("With percentage: " + str(percentage) + " of adversarial training it achieves accuracy of: " + str(acc) + " .")
#     accuracies.append(acc)

# accuracies = [0.5803541597059806, 0.7467423989308386, 0.7783160708319412, 0.9139659204811226, 0.9341797527564317, 0.9513865686602071, 0.9632475776812562, 0.9488807216839291, 0.975609756097561, 0.9667557634480455]
# plt.figure(figsize=(5,5))
# plt.plot(percentages, accuracies, "*-", label='FGSM')
# plt.legend()
# plt.yticks(np.arange(0.4, 1.1, step=0.1))
# plt.xticks(np.arange(0, 1.1, step=0.1))
# plt.title("Accuracy vs Percentage")
# plt.xlabel("Percentage of adversarial images in 50% pertubated training set")
# plt.ylabel("Accuracy")
# plt.savefig('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/Results/Accuracy_vs_Percentage_0.1.png')
    

   ################################# 
# Pick the best situation: 0.1 in training and 0.05 in testing
# Use FGSM attack with epsilon 0.1 in the training phase, then test on 100% FGSM-adversarially pertubated test set

# # 0.1+0.05
# first = [64.80120280654862, 89.05780153691948, 94.13631807550952, 97.22686267958571, 98.14567323755429, 98.44637487470766, 98.42966922819913, 98.64684263280989, 99.26495155362512, 99.3317741396592, 98.52990310725025]

# # 0.15 + 0.1
# second = [37.253591713999334, 58.820581356498494, 61.142666221182765, 61.69395255596392, 64.75108586702305, 70.13030404276645, 77.24690945539592, 81.94119612429, 83.92916805880387, 84.76445038422987, 98.94754426996325]

# # 0.2+0.15
# third = [20.748412963581693, 48.27931840962245, 51.13598396257936, 49.86635482793184, 49.9164717674574, 51.06916137654527, 51.33645172068159, 52.42231874373539, 56.615436017373874, 56.76578683595055, 91.88105579685933]

# # 0.2+0.2
# fourth = [19.011025726695625, 22.55262278650184, 26.962913464751086, 32.626127631139326, 31.256264617440692, 33.21082525893752, 36.28466421650518, 41.179418643501506, 41.11259605746743, 37.48747076511861, 87.9886401603742]
# # 0.25+0.25
# fifth = [23.838957567657868, 28.08219178082192, 30.437687938523222, 31.506849315068493, 31.70731707317073, 28.733711994654193, 20.414300033411294, 17.858336117607752, 20.063481456732376, 16.321416638823923, 79.03441363180755]
# # 0.3+0.3
# sixth = [27.347143334447043, 27.51419979953224, 24.94153023722018, 31.490143668559973, 31.60708319411961, 30.47109923154026, 24.189776144336786, 16.60541262946876, 28.98429669228199, 16.271299699298364, 69.7460741730705]

#############
########### 50% ###############

# percentages = [i/10 for i in range(0, 11)]
# accuracies = []

# for percentage in percentages:
#     # train(epoch_num, model, train_loader, val_loader, 'experiment_2/how_much_adv_help/0.1+0.05/Adversarially_trained_' + str(percentage), percentage, 'fgsm', train_ep)
#     model.load_state_dict(torch.load('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_2/how_much_adv_help/0.3+0.3/Adversarially_trained_' + str(percentage) + '.pth'))
#     model.eval()
#     acc, _ = adv_test(model, device, test_loader, 0.3, 'fgsm', 0.5)
#     print("With percentage: " + str(percentage) + " of adversarial training tested on a 100% perturbed test set it achieves accuracy of: " + str(acc) + " .")
#     accuracies.append(acc * 100)
# print(accuracies)

full_1 = [74.0227196792516, 85.98396257935183, 90.2439024390244, 92.04811226194454, 98.36284664216505, 98.44637487470766, 98.79719345138656, 98.86401603742064, 99.43200801871032, 99.51553625125293, 98.5800200467758]
full_2 = [41.22953558302706, 58.93752088205814, 62.04477113264283, 62.89675910457735, 61.493484797861676, 64.96825927163381, 62.19512195121951, 66.33812228533245, 63.48145673237554, 72.20180420982291, 98.96424991647177]
full_3 = [21.500167056465084, 37.33711994654193, 45.07183427998663, 44.83795522886736, 48.19579017707985, 50.15035081857668, 50.501169395255594, 56.866020715001675, 53.74206481790845, 52.42231874373539, 98.41296358169062]
full_4 = [46.191112596057465, 46.174406949548946, 46.174406949548946, 47.978616772469096, 47.343802205145344, 45.07183427998663, 46.174406949548946, 46.99298362846642, 43.60173738723689, 48.14567323755429, 92.84998329435349]
full_5 = [39.54226528566656, 37.53758770464417, 38.823922485800196, 41.56364851319746, 40.22719679251587, 35.56632141663882, 35.666555295689946, 36.3848980955563, 35.78349482124958, 42.449047778149016, 90.99565653190778]
full_6 = [32.993651854326764, 33.22753090544604, 32.75977280320748, 29.35182091546943, 33.06047444036084, 32.659538924156365, 30.437687938523222, 32.19178082191781, 34.51386568660207, 44.336785833611756, 90.11025726695622]


half_1 = [78.86735716672236, 89.04109589041096, 91.71399933177415, 93.35115268960908, 98.79719345138656, 98.8306047444036, 99.0811894420314, 97.44403608419645, 97.69462078182426, 98.09555629802873, 97.32709655863681]
half_2 = [60.30738389575676, 68.19244904777815, 69.1613765452723, 69.2449047778149, 69.61242900100234, 70.53123955897094, 69.8296024056131, 71.50016705646507, 69.89642499164718, 76.49515536251253, 89.64249916471768]
half_3 = [51.18610090210491, 56.33144002672903, 63.39792849983294, 65.92048112261945, 64.1997995322419, 66.43835616438356, 66.90611426662213, 66.93952555963915, 67.0564650851988, 66.20447711326428, 85.66655529568993]
half_4 = [63.665218843969264, 63.79886401603741, 64.26662211827598, 65.51954560641497, 64.65085198797193, 62.84664216505179, 64.98496491814232, 65.13531573671901, 61.81089208152355, 65.18543267624457, 80.10357500835282]
half_5 = [57.9017707985299, 55.49615770130304, 57.584363514868016, 61.69395255596392, 60.49114600735048, 55.34580688272636, 55.763448045439354, 56.71566989642499, 56.2145005011694, 63.782158369528894, 71.73404610758436]
half_6 = [51.50350818576679, 51.08586702305379, 50.785165385900434, 49.7494153023722, 51.15268960908787, 51.620447711326435, 50.81857667891747, 50.9188105579686, 52.38890745071835, 63.347811560307385, 69.56231206147679]
# percentages = [i for i in range(0, 110, 10)]
# plt.figure(figsize=(5,5))
# plt.plot(percentages, first, "*-", label='0.1, 0.05')
# plt.plot(percentages, second, "*-", label='0.15, 0.1')
# plt.plot(percentages, third, "*-", label='0.2, 0.15')
# plt.plot(percentages, fourth, "*-", label='0.2, 0.2')
# plt.plot(percentages, fifth, "*-", label='0.25, 0.25')
# plt.plot(percentages, sixth, "*-", label='0.3, 0.3')
# plt.legend()
# plt.yticks(np.arange(0, 110, step=10))
# plt.xticks(np.arange(0, 110, step=10))
# plt.title("Accuracy vs Percentage")
# plt.xlabel("Number(%) of adversarial images in training set")
# plt.ylabel("Accuracy(%)")
# plt.savefig('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/experiment_2/how_much_adv_help/accuracy_vs_percentage.png')

########## Experiment 3: Purely train on adversarial dataset, test on clean test set? ############
# adv_train = True
# train(epoch_num, model, train_loader, val_loader, 'Pure_adv_trained', 1)
# # model.load_state_dict(torch.load('/homes/yx3017/Desktop/Individual_project/Individual_project/MedNist_experiment/Individual_project/MedNist_experiment/Pure_adv_trained.pth'))
# model.eval()

# normal_testing(model, device, test_loader)