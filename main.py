import os
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from _operator import truediv
from torchsummary import summary
from natsort import natsorted

from config import *
from dataset import *
from metrics import *
from models import *
from utils import *


def train(model=None, num_epochs=100, train_loader=None, val_loader=None, criterion=None, optimizer=None, dir='/workspace/users/Group-2/', save_name=None):
    print('\nTraining')
    start_time = time.time()

    # Initialize the directories where the checkpoints and logs will be saved.
    check_dir_make(dir + 'Checkpoints/' + save_name)
    check_dir_make(dir + 'Logs/' + save_name)

    # Initialize the csv loggers.
    init_log(dir + 'Logs/' + save_name + '/loss',
             ['epoch', 'loss', 'average_loss'])
    init_log(dir + 'Logs/' + save_name + '/metrics_val',
             ['epoch', 'acc', 'recall', 'prec', 'f1'])

    total_step = len(train_loader)

    for epoch in range(num_epochs):
        average_loss = 0
        for i, (imgs, lbls) in enumerate(train_loader):
            imgs = imgs.to(device).float()
            lbls = lbls.to(device)
            optimizer.zero_grad()

            outputs = model(imgs)

            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            average_loss += loss.item()

        # scheduler.step()     # Activate only when stepLR is being used

        average_loss = truediv(average_loss, total_step)

        train_message = f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Average Loss: {average_loss:.4f}'
        print(train_message)

        write_log(dir + 'Logs/' + save_name + '/loss',
                  [epoch, round(loss.item(), 4), round(average_loss, 4)])

        # cross validation
        if val_loader is not None:
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():

                    model.eval()
                    pred, gt = [], []

                    for imgs_val, lbls_val in val_loader:

                        imgs_val = imgs_val.to(device).float()
                        lbls_val = lbls_val.to(device)

                        # Forward pass
                        outputs_val = torch.max(model(imgs_val), 1)[1]

                        gt.extend(lbls_val.squeeze().cpu().numpy())
                        pred.extend(outputs_val.squeeze().cpu().numpy())

                    gt = np.asarray(gt, np.float32)
                    pred = np.asarray(pred)
                    acc, recall, prec, f1 = calc_metrics(pred, gt)

                    write_log(dir + 'Logs/' + save_name +
                              '/metrics_val', (epoch+1, acc, recall, prec, f1))
                    torch.save(model.state_dict(), dir + 'Checkpoints/' +
                               save_name + '/e' + str(epoch+1) + '.ckpt')

                model.train()

        else:
            torch.save(model.state_dict(), dir + 'Checkpoints/' +
                       save_name + '/e' + str(epoch+1))

    end_time = time.time()
    print(f'Training time: {end_time - start_time}')


def test(model=None, test_loader=None, dir='/workspace/users/Group-2/', save_name=None):
    print('\nTesting')
    start_time = time.time()

    # Initialize the csv logger.
    init_log(dir + 'Logs/' + save_name + '/metrics_test',
             ['epoch', 'acc', 'recall', 'prec', 'f1'])
    ckpts_dir = dir + 'Checkpoints/' + save_name + '/'
    # This makes sure that the ckpts are loaded in a correct order.
    ckpts = natsorted(os.listdir(ckpts_dir))
    accs = []

    for ckpt in ckpts:
        model.load_state_dict(torch.load(ckpts_dir + ckpt))
        model.eval()
        with torch.no_grad():
            probs, pred, gt = None, [], []

            for imgs, lbls in test_loader:
                imgs = imgs.to(device).float()
                lbls = lbls.to(device)
                prob = model(imgs)
                outputs = torch.argmax(prob, dim=1)

                prob = prob.cpu().numpy()
                if probs is None:
                    probs = np.asarray(prob, np.float32)
                else:
                    probs = np.append(probs, prob, axis=0)

                gt.extend(lbls.squeeze().cpu().numpy())
                pred.extend(outputs.squeeze().cpu().numpy())

            gt = np.asarray(gt, np.float32)
            pred = np.asarray(pred)

            acc, recall, prec, f1 = calc_metrics(pred, gt)

            epoch = ckpt.split('e')[1].split('.ckpt')[0]
            write_log(dir + 'Logs/' + save_name + '/metrics_test',
                      (epoch, acc, recall, prec, f1))
            accs.append(acc)

    end_time = time.time()
    print(f'Testing time: {end_time - start_time}')
    print(f'Average Accuracy: {np.mean(accs)} \t Std: {np.std(accs)}')


# This method allows us to test the performance of a single checkpoint.
def test_ckpt(model=None, test_loader=None, dir='/workspace/users/Group-2/', save_name=None, ckpt=None):
    model.load_state_dict(torch.load(
        dir + 'Checkpoints/' + save_name + '/' + ckpt + '.ckpt'))
    model.eval()
    with torch.no_grad():
        probs, pred, gt = None, [], []

        for imgs, lbls in test_loader:
            imgs = imgs.to(device).float()
            lbls = lbls.to(device)
            prob = model(imgs)
            outputs = torch.argmax(prob, dim=1)

            prob = prob.cpu().numpy()
            if probs is None:
                probs = np.asarray(prob, np.float32)
            else:
                probs = np.append(probs, prob, axis=0)

            gt.extend(lbls.squeeze().cpu().numpy())
            pred.extend(outputs.squeeze().cpu().numpy())

        gt = np.asarray(gt, np.float32)
        pred = np.asarray(pred)

        acc, recall, prec, f1 = calc_metrics(pred, gt)
        conf = calc_confusion(pred, gt)

        print(conf)
        print(f'Ckpt: {ckpt} Acc: {acc}')


# This method validates the predictions of the random models against the test set.
def test_random_model(test_data=None, dir='/workspace/users/Group-2/', save_name=None, mode='all'):
    print('\nTesting')
    start_time = time.time()

    check_dir_make(dir + 'Logs/' + save_name)
    init_log(dir + 'Logs/' + save_name + '/metrics_test',
             ['epoch', 'acc', 'recall', 'prec', 'f1'])
    accs = []
    for epoch in range(500):
        _, lbls = test_data[:]
        pred = predict_randomly(mode)
        acc, recall, prec, f1 = calc_metrics(pred, lbls)

        write_log(dir + 'Logs/' + save_name + '/metrics_test',
                  (epoch+1, acc, recall, prec, f1))
        accs.append(acc)

    end_time = time.time()
    print(f'Testing time: {end_time - start_time}')
    print(f'Average Accuracy: {np.mean(accs)} \t Std: {np.std(accs)}')


# This module shows filter and kernel visualizations for a selected model. 
def visualize(model=None, test_loader=None, dir='/workspace/users/Group-2/', save_name=None, ckpt=None):
    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Test the model
    model.load_state_dict(torch.load(
        dir + 'Checkpoints/' + save_name + '/' + ckpt + '.ckpt'))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    # visualize filter
    weight1 = model.conv11.weight.data.cpu().numpy()
    weight2 = model.conv12.weight.data.cpu().numpy()

    print(weight1.shape)
    print(weight2.shape)

    f11 = weight1[0, 0]
    print(f11.shape)
    plt.imshow(f11)
    plt.savefig(dir + 'Logs/' + save_name + '/vis0' + ckpt + '.png')

    fig = plt.figure(figsize=(12, 5))
    # Plot all filter
    for i in range(16):
        axt = fig.add_subplot(4, 4, i+1)
        plt.matshow(weight1[i, 0], fignum=False)
    plt.tight_layout()
    plt.savefig(dir + 'Logs/' + save_name + '/vis1' + ckpt + '.png')

    imgTest = transform_normalize(next(iter(test_loader))[0][0]).cuda()
    imgInter = model.conv11(imgTest.unsqueeze(
        0)).squeeze(0).detach().cpu().numpy()

    fig = plt.figure(figsize=(12, 5))
    # Plot all filter
    for i in range(16):
        axt = fig.add_subplot(4, 4, i+1)
        plt.matshow(imgInter[i], fignum=False)
    plt.tight_layout()
    plt.savefig(dir + 'Logs/' + save_name + '/vis2' + ckpt + '.png')
    

# This module shows feature attribute and layer attribute visualizations for a selected model.     
def vis_integrated(model=None, save_name=None, test_loader=None, type = 0):
     
    from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
    from captum.attr import visualization as viz
    from matplotlib.colors import LinearSegmentedColormap
    transform_normalize = transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]
     )
    
    # Test the model
    model.load_state_dict(torch.load(dir + 'Checkpoints/' + save_name + '/' + ckpt + '.ckpt'))
    model.eval()  
    
    
    
    # Initialize the attribution algorithm with the model
    integrated_gradients = IntegratedGradients(model)
    
    iterx = next(iter(test_loader))
    transformed_img = iterx[0][0].cuda()
    imgTest = transform_normalize(transformed_img).unsqueeze(0)
    
    
    label = iterx[1][0]
    
    # print(imgTest.shape, label)
    # exit(0)
    
    # feature attribution
    if type == 0:  
        
        # Ask the algorithm to attribute our output target to
        attributions_ig = integrated_gradients.attribute(imgTest, target=label, n_steps=10)
        
        # Show the original image for comparison
        _ = viz.visualize_image_attr(None, np.transpose(imgTest.squeeze().cpu().detach().numpy(), (1,2,0)),
                              method="original_image", title="Original Image")
        
        default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                         [(0, '#ffffff'),
                                                          (0.25, '#0000ff'),
                                                          (1, '#0000ff')], N=256)
        
        _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                     np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                     method='heat_map',
                                     cmap=default_cmap,
                                     show_colorbar=True,
                                     sign='positive',
                                     title='Integrated Gradients')
    
    # similar to 0,  but with occlusion
    if type == 1:  
        occlusion = Occlusion(model)
        attributions_occ = occlusion.attribute(imgTest,
                                               target=label,
                                               strides=(3, 8, 8),
                                               sliding_window_shapes=(3,15, 15),
                                               baselines=0)
        
        
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              ["original_image", "heat_map", "heat_map", "masked_image"],
                                              ["all", "positive", "negative", "positive"],
                                              show_colorbar=True,
                                              titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                                              fig_size=(18, 6)
                                             )
    
    #similar to 0,  but with occlusion
    if type == 2:  
        
        layer_gradcam = LayerGradCam(model, model.conv1)
        attributions_lgc = layer_gradcam.attribute(imgTest, target=label)
        
        _ = viz.visualize_image_attr(attributions_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                                     sign="all",
                                     title="Layer 3 Block 1 Conv 2")
        
        
        #additional visualizations
        upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, imgTest.shape[2:])
        print(attributions_lgc.shape)
        print(upsamp_attr_lgc.shape)
        print(imgTest.shape)
        
        _ = viz.visualize_image_attr_multiple(upsamp_attr_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                                              transformed_img.cpu().permute(1,2,0).numpy(),
                                              ["original_image","blended_heat_map","masked_image"],
                                              ["all","positive","positive"],
                                              show_colorbar=True,
                                              titles=["Original", "Positive Attribution", "Masked"],
                                              fig_size=(18, 6))


if __name__ == '__main__':
    # Dataset parameters
    dir = '/workspace/users/Group-2/'
    data_dir = dir + 'Data/IDRID_dataset/'
    batch_size = 16
    shuffle = True
    num_workers = 4

    cross_num = None
    cross_ids_train = [1, 2, 4, 5]
    cross_ids_val = [5]
    norm_per_image = False
    norm_per_dataset = False
    augment = None
    input_size = 64
    tr = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    # Init network
    sel = None 
    
    list_models = ['resnet', 'alexnet', 'vgg','squeezenet', 'densenet', 'inception']

    if sel is not None:
        net, input_size = get_pretrained_models(model_name=list_models[sel], num_classes=5, freeze_prior=False, use_pretrained=False)

        tr = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    else:
        net = CNNBase()
    
    net = net.to(device)


    # Network parameters
    lr = 0.0001
    num_epochs = 100
    optimizer = torch.optim.Adam(net.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    # General parameters
    # The save_name is very important.
    # At the beginning of the training, the according directories will be created based on the save_name.
    # For testing, the checkpoints and logs will be loaded from the according directories.
    # If a save_name is used twice, all of its previous results will be overwritten.
    save_name = 'baseline'
    operation = 2
    ckpt = 'e1'

    # Data loading
    print('Loading dataset')
    start_time_load = time.time()

    # Load train data
    if operation in [0, 2]:
        img_dir = data_dir + 'images/train/'
        lbl_file = data_dir + 'labels/train.csv'
        IDRID_train = Dataset(img_dir, lbl_file, tr, cross_num=cross_num,
                              cross_ids=cross_ids_train, norm_per_image=norm_per_image, norm_per_dataset=norm_per_dataset, augment=augment)
        train_loader = torch.utils.data.DataLoader(
            dataset=IDRID_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        if cross_num is not None:
            IDRID_val = Dataset(img_dir, lbl_file, tr, cross_num=cross_num,
                                cross_ids=cross_ids_val, norm_per_image=norm_per_image, norm_per_dataset=norm_per_dataset, augment=augment)
            val_loader = torch.utils.data.DataLoader(
                dataset=IDRID_val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        else:
            val_loader = None

    # Load test data
    if operation in [1, 2, 4, 5, 6, 7]:
        img_dir = data_dir + 'images/test/'
        lbl_file = data_dir + 'labels/test.csv'
        IDRID_test = Dataset(
            img_dir, lbl_file, tr, norm_per_image=norm_per_image, norm_per_dataset=norm_per_dataset)
        test_loader = torch.utils.data.DataLoader(
            dataset=IDRID_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    end_time_load = time.time()
    print(f'Data load time: {end_time_load - start_time_load}')

    # Operations
    if operation == 0 or operation == 2:
        train(model=net, num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader,
              criterion=criterion, optimizer=optimizer, dir=dir, save_name=save_name)

    if operation == 1 or operation == 2:
        test(model=net, test_loader=test_loader, dir=dir, save_name=save_name)

    if operation == 3:
        summary(net, (3, input_size, input_size))

    if operation == 4:
        test_ckpt(model=net, test_loader=test_loader, dir=dir,
                  save_name=save_name, ckpt=ckpt)

    if operation == 5:
        test_random_model(IDRID_test, dir, save_name, mode='distributed')

    if operation == 6:
        visualize(net, test_loader, dir, save_name, ckpt)
       
    if operation == 7: 
        vis_integrated(net, save_name, test_loader, vis_type)
