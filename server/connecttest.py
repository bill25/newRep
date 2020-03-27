from server.connectionSocket import socketserver
serv = socketserver('127.0.0.1', 9090)

while True:
    msg = serv.recvmsg()
    serv.send_trading_assets(["EURUSD", "GBPUSD"])
