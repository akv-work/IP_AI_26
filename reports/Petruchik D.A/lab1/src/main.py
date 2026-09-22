import numpy as np

c0, c1 = -3, -8

X = np.array([
    [-3, -3],
    [-3, -8],
    [-8, -3],
    [-8, -8]
], dtype=float)

Y_raw = np.array([
    [-3],
    [-8],
    [-8],
    [-3]
], dtype=float)


def normalize(x, xmin, xmax):
    low = min(xmin, xmax)
    high = max(xmin, xmax)
    return (x - low) / (high - low)


def denormalize(y_norm, xmin, xmax):
    low = min(xmin, xmax)
    high = max(xmin, xmax)
    return low + y_norm * (high - low)


Xn = normalize(X, c0, c1)
Y = normalize(Y_raw, c0, c1)


def sigmoid(S):
    return 1.0 / (1.0 + np.exp(-S))


def dsigmoid(y):
    return y * (1.0 - y)

class MLP221:

    def __init__(self, lr=0.5, seed=7):

        rng = np.random.RandomState(seed)

        self.W1 = rng.uniform(-0.1, 0.1, (2, 2))
        self.T1 = rng.uniform(-0.1, 0.1, (1, 2))

        self.W2 = rng.uniform(-0.1, 0.1, (2, 1))
        self.T2 = rng.uniform(-0.1, 0.1, (1, 1))

        self.lr = lr


    def forward(self, x):

        x = x.reshape(1, -1)

        y1 = sigmoid(x @ self.W1 - self.T1)

        y2 = sigmoid(y1 @ self.W2 - self.T2)

        return y1, y2



    def train_step(self, x, e):

        x = x.reshape(1, -1)
        e = e.reshape(1, -1)

        y1, y2 = self.forward(x)


        delta2 = (y2 - e) * dsigmoid(y2)

        delta1 = (delta2 @ self.W2.T) * dsigmoid(y1)


        self.W2 -= self.lr * (y1.T @ delta2)

        self.T2 += self.lr * delta2


        self.W1 -= self.lr * (x.T @ delta1)

        self.T1 += self.lr * delta1


        return 0.5 * np.sum((y2 - e) ** 2)



    def predict(self, x):

        return self.forward(x)[1][0,0]



    def fit(self, X, Y, Ee=0.01, max_epochs=20000):

        for epoch in range(1, max_epochs+1):

            idx = np.random.permutation(len(X))

            Es = sum(
                self.train_step(X[i], Y[i])
                for i in idx
            )


            if Es <= Ee:
                return epoch, Es


        return max_epochs, Es

class SingleLayerPerceptron:


    def __init__(self, lr=0.5, seed=7):

        rng = np.random.RandomState(seed)

        self.W = rng.uniform(-0.1,0.1,(2,1))

        self.T = rng.uniform(-0.1,0.1,(1,1))

        self.lr = lr



    def forward(self,x):

        return sigmoid(
            x.reshape(1,-1) @ self.W - self.T
        )



    def train_step(self,x,e):

        x=x.reshape(1,-1)

        e=e.reshape(1,-1)


        y=self.forward(x)


        delta=(y-e)*dsigmoid(y)


        self.W -= self.lr*(x.T@delta)

        self.T += self.lr*delta


        return 0.5*np.sum((y-e)**2)



    def predict(self,x):

        return self.forward(x)[0,0]



    def fit(self,X,Y,Ee=0.01,max_epochs=20000):

        rng=np.random.RandomState(42)


        for epoch in range(1,max_epochs+1):

            idx=rng.permutation(len(X))


            Es=sum(
                self.train_step(X[i],Y[i])
                for i in idx
            )


            if Es<=Ee:
                return epoch,Es


        return max_epochs,Es


def accuracy(model,Xn,Y,thr=0.5):

    correct=0

    for x,y in zip(Xn,Y):

        pred=model.predict(x)>=thr

        real=int(y[0])

        if pred==bool(real):
            correct+=1


    return correct/len(Xn)


def run_inference(model,A,B,c0,c1,name="Модель"):

    x=normalize(
        np.array([A,B],dtype=float),
        c0,c1
    )


    y_norm=model.predict(x)


    y_hat=denormalize(
        y_norm,
        c0,c1
    )


    nearest = c0 if abs(y_hat-c0)<abs(y_hat-c1) else c1


    print(
        f"[{name}] A={A}, B={B} -> "
        f"ŷ={y_hat:.3f} -> класс {nearest}"
    )


    return y_hat,nearest

if __name__=="__main__":


    mlp=MLP221()

    mlp_epochs,mlp_Es=mlp.fit(Xn,Y)



    slp=SingleLayerPerceptron()

    slp_epochs,slp_Es=slp.fit(Xn,Y)



    print(
        "MLP 2-2-1:",
        mlp_epochs,
        "эп.",
        round(mlp_Es,5),
        "точность:",
        accuracy(mlp,Xn,Y)
    )


    print(
        "SLP:",
        slp_epochs,
        "эп.",
        round(slp_Es,5),
        "точность:",
        accuracy(slp,Xn,Y)
    )



    while True:

        raw=input(
            "Введите A B (пусто - выход): "
        ).strip()


        if not raw:
            break


        A,B=map(float,raw.split())


        run_inference(
            mlp,
            A,
            B,
            c0,
            c1,
            "MLP"
        )


        run_inference(
            slp,
            A,
            B,
            c0,
            c1,
            "SLP"
        )