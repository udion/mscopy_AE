from models import *
from utils import *
from tensorboard_logger import configure, log_value
import os

try:
    os.makedirs('../train_logs')
except OSError:
    pass


E1 = Encoder(n_res_blocks=10)
D1 = Decoder(n_res_blocks=10)
A = AE(E1, D1)
A = A.cuda()

testX, _ = next(testiter)
testiter = iter(testloader)
def eval_model(model):
    X = testX
    print('input looks like ...')
    plt.figure()
    imshow(torchvision.utils.make_grid(X))
    
    X = Variable(X).cuda()
    Y = model(X)
    print('output looks like ...')
    plt.figure()
    imshow2(torchvision.utils.make_grid(Y.data.cpu()))


def train_ae(model, modelName, batchsz):
    ########## logging stuff
    configure('../train_logs/'+modelName+'_bsz{}'.format(batchsz), flush_secs=5)
    print('I configured .. ')
    ########################

    def mynorm2(x):
    m1 = torch.min(x)
    m2 = torch.max(x)
    return (x-m1)/(m2-m1)

    mytransform2 = transforms.Compose(
        [transforms.RandomCrop((41,41)),
        transforms.ToTensor(),
        transforms.Lambda( lambda x : mynorm2(x))])

    trainset = dsets.ImageFolder(root='../sample_dataset/train/',transform=mytransform2)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = dsets.ImageFolder(root='../sample_dataset/test/',transform=mytransform2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    nepoch = 500
    Criterion2 = nn.MSELoss()
    Criterion1 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    loss_track = []
    for eph in range(nepoch):
        
        dataloader = iter(trainloader)
        print('starting epoch {} ...'.format(eph))
        mean_L2_term = 0
        mean_vl3_term = 0
        mean_total_loss = 0
        tot_count = 0

        for i, (X, _) in enumerate(dataloader):
            tot_count += X.size()[0]
            X = Variable(X).cuda()
            optimizer.zero_grad()
            reconX = model(X)
            l2 = Criterion2(reconX, X)

            t1, t2, t3 = vgg19_exc(X)
            rt1, rt2, rt3 = vgg19_exc(reconX)
            # t1 = Variable(t1.data)
            # rt1 = Variable(rt1.data)
            # t2 = Variable(t2.data)
            # rt2 = Variable(rt2.data)
            t3 = Variable(t3.data)
            rt3 = Variable(rt3.data)
            vl3 = Criterion2(rt3, t3)
            
            reconTerm = 10*l2 + vl3
            loss = reconTerm
            loss.backward()
            optimizer.step()

            mean_L2_term += l2.data[0]
            mean_vl3_term += vl3.data[0]
            mean_total_loss += loss.data[0]
            # if i%rec_interval == 0:
            #     loss_track.append(loss.data[0])
            # if i%disp_interval == 0:
            #     print('epoch:{}, iter: {}, L2term:{}, vl3: {}, reconTerm: {}'.format(
            #         eph, i, l2.data[0], vl3.data[0], reconTerm.data[0]))
        
        mean_L2_term /= tot_count
        mean_vl3_term /= tot_count
        mean_total_loss /= tot_count
        log_value('L2_term', mean_L2_term, eph)
        log_value('vl3_term', mean_vl3_term, eph)
        log_value('total_loss', mean_total_term, eph)

        print('epoch:{}, mean_L2term:{}, mean_vl3: {}, mean_reconTerm: {}'.format(eph, mean_L2_term, mean_vl3_term, mean_total_loss))
        save_model(model, modelName)
    return loss_track

