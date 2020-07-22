from gfl.io.eth import ethereum


ethereum.w3_instance().eth.defaultAccount = ethereum.coinbase
ethereum.unlock_account(ethereum.coinbase, "localhost")


if __name__ == "__main__":
    controller = ethereum.contract("controller")
    con_addr = controller.deploy()
    print(con_addr)
    print("-----------------------------")
    storage = ethereum.contract("storage")
    sto_addr = storage.deploy()
    print(sto_addr)