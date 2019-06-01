import matplotlib.pyplot as plt
import numpy as np


Z_DIM = 1
G_HIDDEN = 10
X_DIM = 10
D_HIDDEN = 10

step_size_G = 0.01
step_size_D = 0.01
ITER_NUM = 50000

GRADIENT_CLIP = 0.2
WEIGHT_CLIP = 0.25

TRAIN = False
DRAW_REAL = True

def get_samples(random=True):
    if random:
        x0 = np.random.uniform(0, 1)
        freq = np.random.uniform(1.2, 1.5)
        mult = np.random.uniform(0.5, 0.8)
    else:
        x0 = 0
        freq = 0.2
        mult = 1
    signal = [mult * np.sin(x0+freq*i) for i in range(X_DIM)]
    return np.array(signal)

def ReLU(x):
    return np.maximum(x, 0.)

def dReLU(x):
    return ReLU(x)

def LeakyReLU(x, k=0.2):
    return np.where(x >= 0, x, x * k)

def dLeakyReLU(x, k=0.2):
    return np.where(x >= 0, 1., k)

def Tanh(x):
    return np.tanh(x)

def dTanh(x):
    return 1. - Tanh(x)**2

def Sigmoid(x):
    return 1. / (1. + np.exp(-x))

def dSigmoid(x):
    return Sigmoid(x) * (1. - Sigmoid(x))

def weight_initializer(in_channels, out_channels):
    # scale = np.sqrt(1. / in_channels)
    scale = np.sqrt(2. / (in_channels + out_channels))
    # scale = 0.05
    return np.random.uniform(-scale, scale, (in_channels, out_channels))

class Generator(object):
    def __init__(self):
        self.z = None
        self.w1 = weight_initializer(Z_DIM, G_HIDDEN)
        self.b1 = weight_initializer(1, G_HIDDEN)
        self.x1 = None
        self.w2 = weight_initializer(G_HIDDEN, G_HIDDEN)
        self.b2 = weight_initializer(1, G_HIDDEN)
        self.x2 = None
        self.w3 = weight_initializer(G_HIDDEN, X_DIM)
        self.b3 = weight_initializer(1, X_DIM)
        self.x3 = None
        self.x = None

    def forward(self, inputs):
        self.z = inputs.reshape(1, Z_DIM)
        self.x1 = np.matmul(self.z, self.w1) + self.b1
        self.x1 = ReLU(self.x1)
        self.x2 = np.matmul(self.x1, self.w2) + self.b2
        self.x2 = ReLU(self.x2)
        self.x3 = np.matmul(self.x2, self.w3) + self.b3
        self.x = Tanh(self.x3)
        return self.x

    def backward(self, outputs):
        # Derivative w.r.t. output
        delta = outputs
        delta *= dTanh(self.x)
        # Derivative w.r.t. w3
        d_w3 = np.matmul(np.transpose(self.x2), delta)
        # Derivative w.r.t. b3
        d_b3 = delta.copy()
        # Derivative w.r.t. x2
        delta = np.matmul(delta, np.transpose(self.w3))
        # Update w3
        if np.linalg.norm(d_w3) > GRADIENT_CLIP:
            d_w3 = GRADIENT_CLIP / np.linalg.norm(d_w3) * d_w3
        self.w3 -= step_size_G * d_w3
        self.w3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w3))
        # Update b3
        self.b3 -= step_size_G * d_b3
        self.b3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b3))
        delta *= dReLU(self.x2)
        # Derivative w.r.t. w2
        d_w2 = np.matmul(np.transpose(self.x1), delta)
        # Derivative w.r.t. b2
        d_b2 = delta.copy()
        # Derivative w.r.t. x1
        delta = np.matmul(delta, np.transpose(self.w2))
        # Update w2
        if np.linalg.norm(d_w2) > GRADIENT_CLIP:
            d_w2 = GRADIENT_CLIP / np.linalg.norm(d_w2) * d_w2
        self.w2 -= step_size_G * d_w2
        self.w2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w2))
        # Update b2
        self.b2 -= step_size_G * d_b2
        self.b2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b2))
        delta *= dReLU(self.x1)
        # Derivative w.r.t. w1
        d_w1 = np.matmul(np.transpose(self.z), delta)
        # Derivative w.r.t. b1
        d_b1 = delta.copy()
        # No need to calculate derivative w.r.t. z
        # Update w1
        if np.linalg.norm(d_w1) > GRADIENT_CLIP:
            d_w1 = GRADIENT_CLIP / np.linalg.norm(d_w1) * d_w1
        self.w1 -= step_size_G * d_w1
        self.w1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w1))
        # Update b1
        self.b1 -= step_size_G * d_b1
        self.b1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b1))

class Discriminator(object):
    def __init__(self):
        self.x = None
        self.w1 = weight_initializer(X_DIM, D_HIDDEN)
        self.b1 = weight_initializer(1, D_HIDDEN)
        self.y1 = None
        self.w2 = weight_initializer(D_HIDDEN, D_HIDDEN)
        self.b2 = weight_initializer(1, D_HIDDEN)
        self.y2 = None
        self.w3 = weight_initializer(D_HIDDEN, 1)
        self.b3 = weight_initializer(1, 1)
        self.y3 = None
        self.y = None

    def forward(self, inputs):
        self.x = inputs.reshape(1, X_DIM)
        self.y1 = np.matmul(self.x, self.w1) + self.b1
        self.y1 = LeakyReLU(self.y1)
        self.y2 = np.matmul(self.y1, self.w2) + self.b2
        self.y2 = LeakyReLU(self.y2)
        self.y3 = np.matmul(self.y2, self.w3) + self.b3
        self.y = Sigmoid(self.y3)
        return self.y

    def backward(self, outputs, apply_grads=True):
        # Derivative w.r.t. output
        delta = outputs
        delta *= dSigmoid(self.y)
        # Derivative w.r.t. w3
        d_w3 = np.matmul(np.transpose(self.y2), delta)
        # Derivative w.r.t. b3
        d_b3 = delta.copy()
        # Derivative w.r.t. y2
        delta = np.matmul(delta, np.transpose(self.w3))
        if apply_grads:
            # Update w3
            if np.linalg.norm(d_w3) > GRADIENT_CLIP:
                d_w3 = GRADIENT_CLIP / np.linalg.norm(d_w3) * d_w3
            self.w3 += step_size_D * d_w3
            self.w3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w3))
            # Update b3
            self.b3 += step_size_D * d_b3
            self.b3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b3))
        delta *= dLeakyReLU(self.y2)
        # Derivative w.r.t. w2
        d_w2 = np.matmul(np.transpose(self.y1), delta)
        # Derivative w.r.t. b2
        d_b2 = delta.copy()
        # Derivative w.r.t. y1
        delta = np.matmul(delta, np.transpose(self.w2))
        if apply_grads:
            # Update w2
            if np.linalg.norm(d_w2) > GRADIENT_CLIP:
                d_w2 = GRADIENT_CLIP / np.linalg.norm(d_w2) * d_w2
            self.w2 += step_size_D * d_w2
            self.w2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w2))
            # Update b2
            self.b2 += step_size_D * d_b2
            self.b2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b2))
        delta *= dLeakyReLU(self.y1)
        # Derivative w.r.t. w1
        d_w1 = np.matmul(np.transpose(self.x), delta)
        # Derivative w.r.t. b1
        d_b1 = delta.copy()
        # Derivative w.r.t. x
        delta = np.matmul(delta, np.transpose(self.w1))
        # Update w1
        if apply_grads:
            if np.linalg.norm(d_w1) > GRADIENT_CLIP:
                d_w1 = GRADIENT_CLIP / np.linalg.norm(d_w1) * d_w1
            self.w1 += step_size_D * d_w1
            self.w1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w1))
            # Update b1
            self.b1 += step_size_D * d_b1
            self.b1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b1))
        return delta

class LossFunc(object):
    def __init__(self):
        self.logit = None
        self.label = None

    def forward(self, logit, label):
        if logit[0, 0] < 1e-7:
            logit[0, 0] = 1e-7
        if 1. - logit[0, 0] < 1e-7:
            logit[0, 0] = 1. - 1e-7
        self.logit = logit
        self.label = label
        return - (label * np.log(logit) + (1-label) * np.log(1-logit))

    def backward(self):
        return (1-self.label) / (1-self.logit) - self.label / self.logit

def report(var, name):
    print('{}  {}'.format(name, var))

G = Generator()
D = Discriminator()
criterion = LossFunc()

real_label = 1
fake_label = 0

if TRAIN:
    D_loss = []
    G_loss = []
    for itr in range(ITER_NUM):
        # Update D with real data
        x_real = get_samples(True)
        y_real = D.forward(x_real)
        loss_D_r = criterion.forward(y_real, real_label)
        d_loss_D = criterion.backward()
        D.backward(d_loss_D)

        # Update D with fake data
        z_noise = np.random.randn(Z_DIM)
        x_fake = G.forward(z_noise)
        y_fake = D.forward(x_fake)
        loss_D_f = criterion.forward(y_fake, fake_label)
        d_loss_D = criterion.backward()
        D.backward(d_loss_D)

        # Update G with fake data
        y_fake_r = D.forward(x_fake)
        loss_G = criterion.forward(y_fake_r, real_label)
        d_loss_G = D.backward(loss_G, apply_grads=False)
        G.backward(d_loss_G)
        loss_D = loss_D_r + loss_D_f
        if itr % 100 == 0:
            # print('{}  {}'.format(loss_D.item((0,0)), loss_G.item((0,0))))
            # print('{}  {}'.format(y_real.item((0,0)), y_fake.item((0,0))))
            print('{}  {}  {}'.format(loss_D_r.item((0, 0)), loss_D_f.item((0, 0)), loss_G.item((0, 0))))
        D_loss.append(loss_D.item((0,0)))
        G_loss.append(loss_G.item((0,0)))
    np.save('G_w1.npy', G.w1)
    np.save('G_w2.npy', G.w2)
    np.save('G_w3.npy', G.w3)
    np.save('G_b1.npy', G.b1)
    np.save('G_b2.npy', G.b2)
    np.save('G_b3.npy', G.b3)
    np.save('G_loss.npy', np.array(G_loss))
    np.save('D_w1.npy', D.w1)
    np.save('D_w2.npy', D.w2)
    np.save('D_w3.npy', D.w3)
    np.save('D_b1.npy', D.b1)
    np.save('D_b2.npy', D.b2)
    np.save('D_b3.npy', D.b3)
    np.save('D_loss.npy', np.array(D_loss))
else:
    G.w1 = np.load('G_w1.npy')
    G.w2 = np.load('G_w2.npy')
    G.w3 = np.load('G_w3.npy')
    G.b1 = np.load('G_b1.npy')
    G.b2 = np.load('G_b2.npy')
    G.b3 = np.load('G_b3.npy')
    D.w1 = np.load('D_w1.npy')
    D.w2 = np.load('D_w2.npy')
    D.w3 = np.load('D_w3.npy')
    D.b1 = np.load('D_b1.npy')
    D.b2 = np.load('D_b2.npy')
    D.b3 = np.load('D_b3.npy')

    x_axis = np.linspace(0, 10, 10)
    for i in range(50):
        if not DRAW_REAL:
            z_noise = np.random.randn(Z_DIM)
            x_fake = G.forward(z_noise)
            y_fake = D.forward(x_fake)
            loss_D = criterion.forward(y_fake, real_label)
            plt.plot(x_axis, x_fake.reshape(X_DIM))
        else:
            x_real = get_samples(True)
            plt.plot(x_axis, x_real)
    plt.ylim((-1, 1))
    plt.show()
