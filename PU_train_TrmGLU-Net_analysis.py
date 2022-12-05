import os
import time


import numpy as np

import torch

import scipy.io as sio
from sklearn.decomposition import PCA
from uformer_nofusion import Uformer

from sklearn.metrics import cohen_kappa_score

import datetime

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     #random.seed(seed)
     torch.backends.cudnn.deterministic = True 

def run(dataname = 'IP', data_path = './image', slic_path = './slic_optimal.npy',segments=100, augment = True, save_path = './', model_name= 'Uformer'):
    gpu_id = 4
    experiment_num=1

    if dataname == 'IP':
        categories=16
    elif dataname == 'PU':
        categories = 9
    elif dataname == 'SA':
        categories = 16
    elif dataname == 'HU':
        categories = 15

    Experiment_result = np.zeros([categories + 5, experiment_num + 2])

    for iter_num in range (experiment_num):
        random_num=iter_num
        if dataname == 'IP':
            train_epoch = 150
            setup_seed(123456789+random_num)
        elif dataname == 'PU':
            train_epoch = 260
            setup_seed(123456789+random_num)
        elif dataname == 'SA':
            train_epoch = 150
            setup_seed(123456789+random_num)
        elif dataname == 'HU':
            train_epoch = 300
            setup_seed(123456789+random_num)



        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # choose GPU:0

        ######read image#########
        if dataname == 'IP':
            file_name = os.path.join(data_path, 'Indian_pines_corrected.mat')
            image1 = sio.loadmat(file_name)
            image1 = image1[list(image1)[-1]][:144,:144,:]
            image1=image1.reshape(-1,200)
            pca=PCA(3)
            y=pca.fit(image1)
            image1=y.transform(image1)
            m,n=image1.max(),image1.min()
            image1=(image1-n)/(m-n)
            image1 = image1.reshape(144,144,-1)
        elif dataname == 'PU':
            file_name = os.path.join(data_path, 'PaviaU.mat')
            image1 = sio.loadmat(file_name)
            image1 = image1[list(image1)[-1]]
            image1=image1.reshape(-1,103)
            #pca=PCA(3)
            #y=pca.fit(image1)
            #image1=y.transform(image1)
            m,n=image1.max(),image1.min()
            image1=(image1-n)/(m-n)
            image1 = image1.reshape(610,340,-1)
            image = np.zeros((640,384,103),dtype = np.float32)
            for i in range(103):
                temp = np.pad(image1[:,:,i],((0,30),(0,44)),'symmetric')
                image[:,:,i] = temp
            image1 = image
        elif dataname == 'SA':
            file_name = os.path.join(data_path, 'Salinas_corrected.mat')
            image1 = sio.loadmat(file_name)
            image1 = image1[list(image1)[-1]]
            image1=image1.reshape(-1,204)
            pca=PCA(3)
            y=pca.fit(image1)
            image1=y.transform(image1)
            m,n=image1.max(),image1.min()
            image1=(image1-n)/(m-n)
            image1 = image1.reshape(512,217,-1)
            image = np.zeros((512,256,3),dtype = np.float32)
            for i in range(3):
                temp = np.pad(image1[:,:,i],((0,0),(0,39)),'symmetric')
                image[:,:,i] = temp
            image1 = image
        elif dataname == 'HU':
            file_name = os.path.join(data_path, 'DFC2013_Houston.mat')
            image1 = sio.loadmat(file_name)
            image1 = image1[list(image1)[-1]]
            image1 = image1.reshape(-1, 144)
            pca = PCA(3)
            y = pca.fit(image1)
            image1 = y.transform(image1)
            m, n = image1.max(), image1.min()
            image1 = (image1 - n) / (m - n)
            image1 = image1.reshape(349, 1905, -1)
            image = np.zeros((350, 1905, 3), dtype=np.float32)
            for i in range(3):
                temp = np.pad(image1[:, :, i], ((0, 1), (0, 0)), 'symmetric')
                image[:, :, i] = temp
            image1 = image



        '''train init'''
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        tensor = image1.transpose((2, 0, 1))
        tensor = tensor.astype(np.float32) / image1.max()#255.0
        tensor = tensor[np.newaxis, :, :, :]
        tensor = torch.from_numpy(tensor).to(device)
        #tensor = torch.unsqueeze(tensor, 0)

        ######read gt#########
        if dataname == 'IP':
            file_name = os.path.join(data_path, 'Indian_pines_gt.mat')
            gt = sio.loadmat(file_name)
            gt = gt[list(gt)[-1]][:144,:144]
            train_gt = np.ones(gt.shape,dtype = np.uint8)*255
            gt1 = gt.reshape(-1)
            train_gt = train_gt.reshape(-1)
            for i in range(gt1.max()):
                temp = np.squeeze(np.argwhere(gt1==(i+1)))
                np.random.seed(123456789)
                np.random.shuffle(temp)
                train_gt[temp[:10]] = i
            train_gt = train_gt.reshape(144,144)
            if augment:
                mask = np.load(slic_path)
                mask = mask[:144,:144]

                train_gt = get_superpixel_gt(train_gt, mask)
        elif dataname == 'PU':
            file_name = os.path.join(data_path, 'PaviaU_gt.mat')
            gt = sio.loadmat(file_name)
            gt = gt[list(gt)[-1]]
            train_gt = np.ones(gt.shape,dtype = np.uint8)*255
            gt1 = gt.reshape(-1)
            train_gt = train_gt.reshape(-1)
            for i in range(gt1.max()):
                temp = np.squeeze(np.argwhere(gt1==(i+1)))
                np.random.seed(123456789)
                np.random.shuffle(temp)
                train_gt[temp[:20]] = i
            train_gt = np.pad(train_gt.reshape(610, 340), ((0, 30), (0, 44)), 'symmetric')
            if augment:

                mask = np.load(slic_path)
                mask = np.pad(mask,((0,30),(0,44)),'symmetric')
                train_gt = get_superpixel_gt(train_gt, mask)
        elif dataname == 'SA':
            file_name = os.path.join(data_path, 'Salinas_gt.mat')
            gt = sio.loadmat(file_name)
            gt = gt[list(gt)[-1]]
            train_gt = np.ones(gt.shape,dtype = np.uint8)*255
            gt1 = gt.reshape(-1)
            train_gt = train_gt.reshape(-1)
            for i in range(gt1.max()):
                temp = np.squeeze(np.argwhere(gt1==(i+1)))
                np.random.seed(123456789)#+random_num
                np.random.shuffle(temp)
                train_gt[temp[:10]] = i#np.uint8(len(temp)*0.01)
            if augment:
                train_gt = np.pad(train_gt.reshape(512,217),((0,0),(0,39)),'symmetric')
                mask = np.load(slic_path)
                mask = np.pad(mask,((0,0),(0,39)),'symmetric')
                train_gt = get_superpixel_gt(train_gt, mask)
        elif dataname == 'HU':
            file_name = os.path.join(data_path, 'DFC2013_Houston_gt.mat')
            gt = sio.loadmat(file_name)
            gt = gt[list(gt)[-1]]
            train_gt = np.ones(gt.shape,dtype = np.uint8)*255
            gt1 = gt.reshape(-1)
            train_gt = train_gt.reshape(-1)
            for i in range(gt1.max()):
                temp = np.squeeze(np.argwhere(gt1==(i+1)))
                np.random.seed(123456789)#+random_num
                np.random.shuffle(temp)
                train_gt[temp[:20]] = i#np.uint8(len(temp)*0.01)
            if augment:
                train_gt = np.pad(train_gt.reshape(349,1905),((0,1),(0,0)),'symmetric')
                mask = np.load(slic_path)
                mask = np.pad(mask,((0,1),(0,0)),'symmetric')
                train_gt = get_superpixel_gt(train_gt, mask)



        train_gt = train_gt.reshape(-1)
        train_gt = torch.from_numpy(train_gt).long()
        train_gt = train_gt.to(device)

        mod_dim2 = 0
        if dataname == 'IP':
            mod_dim2 =16
            model = Uformer(
                dim = 64,           # initial dimensions after input projection, which increases by 2x each stage
                stages = 2,         # number of stages
                num_blocks = 1,     # number of transformer blocks per stage
                window_size = 18,   # set window size (along one side) for which to do the attention within
                dim_head = 64,
                heads = 8,
                ff_mult = 4,
                output_channels = mod_dim2,
                input_channels=3).to(device)
        elif dataname == 'PU':
            mod_dim2 =9
            model = Uformer(
                dim = 64,           # initial dimensions after input projection, which increases by 2x each stage
                stages = 2,         # number of stages
                num_blocks = 1,     # number of transformer blocks per stage
                window_size = 4,   # set window size (along one side) for which to do the attention within
                dim_head = 64,
                heads = 8,
                ff_mult = 4,
                output_channels = mod_dim2,
                input_channels=103).to(device)
        elif dataname == 'SA':
            mod_dim2 =16
            model = Uformer(
                dim = 64,           # initial dimensions after input projection, which increases by 2x each stage
                stages = 2,         # number of stages
                num_blocks = 1,     # number of transformer blocks per stage
                window_size = 8,   # set window size (along one side) for which to do the attention within
                dim_head = 64,
                heads = 8,
                ff_mult = 4,
                output_channels = mod_dim2,
                input_channels=3).to(device)
        elif dataname == 'HU':
            mod_dim2 =15
            model = Uformer(
                dim = 64,           # initial dimensions after input projection, which increases by 2x each stage
                stages = 2,         # number of stages
                num_blocks = 1,     # number of transformer blocks per stage
                window_size = 4,   # set window size (along one side) for which to do the attention within
                dim_head = 64,
                heads = 8,
                ff_mult = 4,
                output_channels = mod_dim2,
                input_channels=3).to(device)


        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)


        '''train loop'''
        model.train()
        epoch = 0
        train_time1 = time.time()
        for batch_idx in range(train_epoch):
            '''forward'''
            optimizer.zero_grad()
            output = model(tensor)[0]
            output = torch.squeeze(output)
            output = output.permute(1, 2, 0).view(-1, mod_dim2)
            loss = criterion(output, train_gt)


            loss.backward()
            optimizer.step()
            epoch += 1
            #print('epoch:%3d, Loss:%f', batch_idx, loss.item())

            if dataname == 'IP':
                temp = np.argwhere(gt1>0)
                pred = torch.argmax(output, 1)
                y_pred = pred.data.cpu().numpy()

                y_pred_map = y_pred.reshape(-1) + 1

                y_pred = y_pred_map[temp]

                y_true = gt1[temp]
                oa = y_pred==y_true
                oa = oa.sum()/len(y_true)
                print('epoch:', batch_idx, 'OA:', oa, 'loss:' + str(loss.item()))


            elif dataname == 'PU':
                temp = np.argwhere(gt1>0)
                pred = torch.argmax(output, 1)

                y_pred = pred.data.cpu().numpy()
                y_pred = y_pred.reshape(640,384)
                y_pred = y_pred[:610,:340]

                y_pred_map = y_pred.reshape(-1) + 1
                #generate_png(gt1, y_pred, dataname, 610, 340,segments)
                y_pred = y_pred_map[temp]

                y_true = gt1[temp]
                oa = y_pred==y_true
                oa = oa.sum()/len(y_true)
                print('epoch:', batch_idx, 'OA:', oa, 'loss:' + str(loss.item()))
            elif dataname == 'SA':
                temp = np.argwhere(gt1>0)
                pred = torch.argmax(output, 1)

                y_pred = pred.data.cpu().numpy()
                y_pred = y_pred.reshape(512,256)
                y_pred = y_pred[:512,:217]

                y_pred_map = y_pred.reshape(-1) + 1
                #generate_png(gt1, y_pred, dataname, 512, 217,segments)
                y_pred = y_pred_map[temp]

                y_true = gt1[temp]
                oa = y_pred==y_true
                oa = oa.sum()/len(y_true)
                print('epoch:', batch_idx, 'OA:', oa, 'loss:' + str(loss.item()))
            elif dataname == 'HU':
                temp = np.argwhere(gt1>0)
                pred = torch.argmax(output, 1)

                y_pred = pred.data.cpu().numpy()
                y_pred = y_pred.reshape(350,1905)
                y_pred = y_pred[:349,:1905]

                y_pred_map = y_pred.reshape(-1) + 1
                #generate_png(gt1, y_pred, dataname, 610, 340,segments)
                y_pred = y_pred_map[temp]

                #y_pred = y_pred.reshape(-1)[temp] + 1
                y_true = gt1[temp]
                oa = y_pred==y_true
                oa = oa.sum()/len(y_true)
                print('epoch:', batch_idx, 'OA:', oa, 'loss:' + str(loss.item()))


            del y_pred

        train_time2 = time.time()

        tes_time1 = time.time()
        model.eval()
        output = model(tensor)[0]
        output = torch.squeeze(output)
        output = output.permute(1, 2, 0).view(-1, mod_dim2)
        tes_time2 = time.time()

        temp = np.argwhere(gt1 > 0)
        pred = torch.argmax(output, 1)
        y_pred = pred.data.cpu().numpy()

        if dataname == 'IP':

            y_pred_map = y_pred.reshape(-1)
            #generate_png(gt1, y_pred_map, dataname, 144, 144, segments)
            y_pred = y_pred_map[temp]
            y_true = gt1[temp]
            oa = y_pred == y_true
            oa = oa.sum() / len(y_true)



        elif dataname == 'PU':

            y_pred = y_pred.reshape(640, 384)
            y_pred = y_pred[:610, :340]
            y_pred_map = y_pred.reshape(-1)
            #generate_png(gt1, y_pred_map, dataname, 610, 340,segments)
            y_pred = y_pred_map[temp]

            y_true = gt1[temp]
            oa = y_pred == y_true
            oa = oa.sum() / len(y_true)

        elif dataname == 'SA':

            y_pred = y_pred.reshape(512, 256)
            y_pred = y_pred[:512, :217]

            y_pred_map = y_pred.reshape(-1)
            #generate_png(gt1, y_pred_map, dataname, 512, 217,segments)
            y_pred = y_pred_map[temp]

            y_true = gt1[temp]
            oa = y_pred == y_true
            oa = oa.sum() / len(y_true)

        elif dataname == 'HU':

            y_pred = y_pred.reshape(350, 1905)
            y_pred = y_pred[:349, :1905]

            y_pred_map = y_pred.reshape(-1)
            #generate_png(gt1, y_pred_map, dataname, 610, 340,segments)
            y_pred = y_pred_map[temp]

            # y_pred = y_pred.reshape(-1)[temp] + 1
            y_true = gt1[temp]
            oa = y_pred == y_true
            oa = oa.sum() / len(y_true)

        print('Experiment {}，Testing set OA={}'.format(iter_num, np.mean(y_true == y_pred)))

        num_tes = np.zeros([categories])
        num_tes_pred = np.zeros([categories])

        for k in y_true:
            num_tes[int(k)-1]+=1# class index start from 0
        for j in range(y_true.shape[0]):
            if y_true[j]==y_pred[j]:
                num_tes_pred[int(y_true[j])-1]+=1

        Acc = num_tes_pred / num_tes * 100

        Experiment_result[0, iter_num] = np.mean(y_true == y_pred) * 100  # OA
        Experiment_result[1, iter_num] = np.mean(Acc)  # AA
        Experiment_result[2, iter_num] = cohen_kappa_score(y_true, y_pred) * 100  # Kappa
        Experiment_result[3, iter_num] = train_time2 - train_time1
        Experiment_result[4, iter_num] = tes_time2 - tes_time1
        Experiment_result[5:, iter_num] = Acc

        print('Experiment {}，Testing set AA={}'.format(iter_num, np.mean(Acc)))
        print('Experiment {}，Testing set Kappa={}'.format(iter_num, cohen_kappa_score(y_true, y_pred)))
        for i in range(categories - 1):
            print('Class_{}: accuracy {:.4f}.'.format(i + 1, Acc[i]))

        print('One time training cost {:.4f} secs'.format(train_time2 - train_time1))
        print('One time testing cost {:.4f} secs'.format(tes_time2 - tes_time1))

        del tensor, train_gt, output
        del y_pred, y_true

        torch.cuda.empty_cache()


        print('########### Experiment {}，Model assessment Finished！ ###########'.format(iter_num))

        ########## mean value & standard deviation #############

    Experiment_result[:, -2] = np.mean(Experiment_result[:, 0:-2], axis=1)  # 计算均值
    Experiment_result[:, -1] = np.std(Experiment_result[:, 0:-2], axis=1)  # 计算平均差

    print('OA_std={}'.format(Experiment_result[0, -1]))
    print('AA_std={}'.format(Experiment_result[1, -1]))
    print('Kappa_std={}'.format(Experiment_result[2, -1]))
    print('time training cost_std{:.4f} secs'.format(Experiment_result[3, -1]))
    print('time testing cost_std{:.4f} secs'.format(Experiment_result[4, -1]))
    for i in range(Experiment_result.shape[0]):
        if i > 4:
            print('Class_{}: accuracy_std {:.4f}.'.format(i - 4, Experiment_result[i, -1]))  # 均差

    day = datetime.datetime.now()
    day_str = day.strftime('%m_%d_%H_%M')

    if augment:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        slic = os.path.basename(slic_path)[:-4]
    else: slic=dataname+'_slic_0'

    f = open('./record/' + str(day_str) + '_' + dataname + '_' + model_name + '_' + str(slic) + '.txt', 'w')
    for i in range(Experiment_result.shape[0]):
        f.write(str(i + 1) + ':' + str(Experiment_result[i, -2]) + '+/-' + str(Experiment_result[i, -1]) + '\n')
    for i in range(Experiment_result.shape[1] - 2):
        f.write('Experiment_num' + str(i) + '_OA:' + str(Experiment_result[0, i]) + '\n')
    f.close()



def get_superpixel_gt(gt, mask):
    m, n = gt.shape
    gt_aug = gt.copy()
    for i in range(m):
        for j in range(n):
            if gt[i,j] != 255:
                index = mask[i,j]
                gt_aug[mask==index] = gt[i,j]
    return gt_aug
       


if __name__ == '__main__':
    path = './slic_optimal/PU'
    #slice_n=[0,100,500,1000,2000,5000,10000]
    slice_n=[1200]

    for n in slice_n:
        print('processing ' + str(n))
        if n == 0:
            run(dataname='PU', data_path='/HSI_data', slic_path=None, segments=0,augment=False, save_path='./PU_results_augment', model_name='Uformer')
        else:
            run(dataname='PU', data_path='/HSI_data', slic_path=path + '/PU_slic_' + str(n) +'.npy', segments=n,augment=True, save_path='./PU_results_augment', model_name='Uformer')

        '''
        elif 'PU' in n:
            for i in range(10):
                run(dataname='PU', data_path='D:/HSI_data', slic_path=path + '/' + file,
                    augment=True, save_path='./PU_resultaugment', model_name='Uformer', random_num=i)
                print(i)
        elif 'SA' in n:
            for i in range(10):
                run(dataname='SA', data_path='D:/HSI_data', slic_path=path + '/' + file,
                    augment=True, save_path='./SA_resultaugment', model_name='Uformer', random_num=i)
                print(i)
        elif 'HU' in n:
            for i in range(10):
                run(dataname='HU', data_path='D:/HSI_data', slic_path=path + '/' + file,
                    augment=True, save_path='./HU_resultaugment', model_name='Uformer', random_num=i)
                print(i)
        '''

    
    