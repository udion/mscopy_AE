from models import *
from utils import *

testiter = iter(testloader)

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

testX, _ = next(testiter)
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

def train(model, rec_interval=2, disp_interval=20, eval_interval=1):
    nepoch = 100
    Criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_track = []
    for eph in range(nepoch):
        dataloader = iter(trainloader)
        print('starting epoch {} ...'.format(eph))
        for i, (X, _) in enumerate(dataloader):
            X = Variable(X).cuda()
            optimizer.zero_grad()
            reconX = model(X)
            KLTerm = latent_loss(model.z_mean, model.z_sigma)
            reconTerm = Criterion(reconX, X)
            loss = reconTerm + KLTerm
            loss.backward()
            optimizer.step()
            
            if i%rec_interval == 0:
                loss_track.append(loss.data[0])
            if i%disp_interval == 0:
                print('epoch : {}, iter : {}, KLterm : {}, reconTerm : {}, totalLoss : {}'.format(eph, i, KLTerm.data[0], reconTerm.data[0], loss.data[0]))
        
        if eph%eval_interval == 0:
            print('after epoch {} ...'.format(eph))
            eval_model(model)
    return loss_track

