from models import *
from utils import *

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


def train_ae(model, rec_interval=2, disp_interval=20, eval_interval=1):
    nepoch = 500
    Criterion2 = nn.MSELoss()
    Criterion1 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    loss_track = []
    for eph in range(nepoch):
        dataloader = iter(trainloader)
        print('starting epoch {} ...'.format(eph))
        for i, (X, _) in enumerate(dataloader):
            X = Variable(X).cuda()
            optimizer.zero_grad()
            reconX = model(X)
            l2 = Criterion2(reconX, X)
            l1 = Criterion1(reconX, X)
            
            t1, t2, t3 = vgg19_exc(X)
            rt1, rt2, rt3 = vgg19_exc(reconX)
            t1 = Variable(t1.data)
            rt1 = Variable(rt1.data)
            t2 = Variable(t2.data)
            rt2 = Variable(rt2.data)
            t3 = Variable(t3.data)
            rt3 = Variable(rt3.data)
            
            reconTerm = l2+l1
            loss = reconTerm
            loss.backward()
            optimizer.step()
            
            if i%rec_interval == 0:
                loss_track.append(loss.data[0])
            if i%disp_interval == 0:
                print('epoch : {}, iter : {}, L2term : {}, L1term : {}, reconTerm : {}, totalLoss : {}'.format(
                    eph, i, l2.data[0], l1.data[0], reconTerm.data[0], loss.data[0]))
        save_model(model, 'AE_VGGFeatX1.pth')
    return loss_track

