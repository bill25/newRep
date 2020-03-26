from server.connectionSocket import socketserver
serv = socketserver('127.0.0.1', 9090)
serv.send_trading_assets(["EURUSD","GBPUSD"])
while True:
    msg = serv.recvmsg()