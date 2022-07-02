import json
from typing import AnyStr

import ecies
from eth_account.messages import encode_defunct
from eth_keys import keys
from web3 import Web3

w3 = Web3()


__global_node: "GflNode"


def check_empty(s: str, name: str):
    if not isinstance(s, str):
        raise ValueError(f"{name}({s}) is not instance of str")
    if s is None:
        raise ValueError(f"{name} cannot not be None")
    if s == "":
        raise ValueError(f"{name} cannot be empty str")


def _set_global_node(node: "GflNode"):
    global __global_node
    check_empty(node.address, "address")
    check_empty(node.pub_key, "pub_key")
    check_empty(node.priv_key, "priv_key")
    __global_node = node


def _get_global_node() -> "GflNode":
    global __global_node
    return __global_node


class GflNode(object):

    def __init__(self, address, pub_key, priv_key=None):
        super(GflNode, self).__init__()
        self.__address = address
        self.__pub_key = pub_key
        self.__priv_key = priv_key

    @property
    def address(self):
        return self.__address

    @property
    def pub_key(self):
        return self.__pub_key

    @property
    def priv_key(self):
        return self.__priv_key

    def sign(self, message: AnyStr) -> str:
        """

        :param message:
        :return:
        """
        if type(message) == str:
            message = message.encode("utf8")
        if type(message) != bytes:
            raise TypeError("message only support str or bytes.")
        encoded_message = encode_defunct(hexstr=message.hex())
        signed_message = w3.eth.account.sign_message(encoded_message, self.__priv_key)
        return signed_message.signature.hex()

    def recover(self, message: AnyStr, signature: str) -> str:
        """
        Get the address of the manager that signed the given message.

        :param message: the message that was signed
        :param signature: the signature of the message
        :return: the address of the manager
        """
        if type(message) == str:
            message = message.encode("utf8")
        if type(message) != bytes:
            raise TypeError("message only support str or bytes.")
        encoded_message = encode_defunct(message)
        return w3.eth.account.recover_message(encoded_message, signature=signature)

    def verify(self, message: AnyStr, signature: str, source_address: str) -> bool:
        """
        Verify whether the message is signed by source address

        :param message: the message that was signed
        :param signature: the signature of the message
        :param source_address: the message sent from
        :return: True or False
        """
        rec_addr = self.recover(message, signature)
        return rec_addr[2:].lower() == source_address.lower()

    def encrypt(self, plain: AnyStr) -> bytes:
        """
        Encrypt with receiver's public key

        :param plain: data to encrypt
        :return: encrypted data
        """
        if type(plain) == str:
            plain = plain.encode("utf8")
        if type(plain) != bytes:
            raise TypeError("message only support str or bytes.")
        cipher = ecies.encrypt(self.__pub_key, plain)
        return cipher

    def decrypt(self, cipher: bytes) -> bytes:
        """
        Decrypt with private key

        :param cipher:
        :return:
        """
        if type(cipher) != bytes:
            raise TypeError("cipher only support bytes.")
        return ecies.decrypt(self.__priv_key, cipher)

    def as_alobal(self):
        _set_global_node(self)

    @classmethod
    def global_instance(cls):
        return _get_global_node()

    @classmethod
    def new_node(cls):
        account = w3.eth.account.create()
        priv_key = keys.PrivateKey(account.key)
        pub_key = priv_key.public_key
        return GflNode(account.address[2:],
                       pub_key.to_hex()[2:],
                       priv_key.to_hex()[2:])

    @classmethod
    def load_node(cls, path):
        with open(path, "r") as f:
            keyjson = json.loads(f.read())
            return GflNode(keyjson.get("address", None),
                           keyjson.get("pub_key", None),
                           keyjson.get("priv_key", None))

    @classmethod
    def save_node(cls, node, path):
        node.save(path)

    def save(self, path):
        with open(path, "w") as f:
            f.write(json.dumps({
                "address": self.__address,
                "pub_key": self.__pub_key,
                "priv_key": self.__priv_key
            }, indent=4))


GflNode.new_node().as_alobal()
