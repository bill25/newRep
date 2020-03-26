import socket, numpy as np
from sklearn.linear_model import LinearRegression


class socketserver:
    def __init__(self, address='', port=9090):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.address = address
        self.port = port
        self.sock.bind((self.address, self.port))
        self.cummdata = ''
        self.assets=''

    def calcregr(self,msg=''):
        chartdata = np.fromstring(msg, dtype=float, sep=' ')
        Y = np.array(chartdata).reshape(-1, 1)
        X = np.array(np.arange(len(chartdata))).reshape(-1, 1)

        lr = LinearRegression()
        lr.fit(X, Y)
        Y_pred = lr.predict(X)
        type(Y_pred)
        P = Y_pred.astype(str).item(-1) + ' ' + Y_pred.astype(str).item(0)
        print(P)
        return str(P)


    def recvmsg(self):
        self.sock.listen(1)
        self.conn, self.addr = self.sock.accept()
        print('connected to', self.addr)
        self.cummdata = ''

        while True:
            data = self.conn.recv(10000)
            print(data)
            self.cummdata += data.decode("utf-8")
            if not data:
                break
            self.conn.send(bytes(self.calcregr(self.cummdata), "utf-8"))
            return self.cummdata

    def send_trading_assets(self,asstes):
        self.sock.listen(1)
        self.conn, self.addr = self.sock.accept()
        print('connected to', self.addr)
        assets = ''
        self.conn.send(bytes(assets.join(asstes), "utf-8"))
        print((assets.join(asstes)))


    def __del__(self):
        self.sock.close()





