import json
import os
from typing import AnyStr, NoReturn

from web3 import Web3
import ecies
from eth_account.messages import encode_defunct
from eth_keys import keys

from gfl.conf import GflConf
from gfl.utils import PathUtils

w3 = Web3()


class GflNodeMetadata(type):

    @property
    def address(cls):
        return cls._GflNode__address

    @property
    def pub_key(cls):
        return cls._GflNode__pub_key

    @property
    def priv_key(cls):
        return cls._GflNode_priv_key


class GflNode(object, metaclass=GflNodeMetadata):
    default_node = None
    standalone_nodes = {}

    __address = None
    __pub_key = None
    __priv_key = None

    def __init__(self):
        """
        """
        super(GflNode, self).__init__()

    @classmethod
    def init_node(cls) -> NoReturn:
        """
        初始化GFL节点

        :return:
        """
        cls.__new_node()
        key_dir = PathUtils.join(GflConf.home_dir, "key")
        os.makedirs(key_dir, exist_ok=True)
        key_file = PathUtils.join(key_dir, "key.json")
        cls.__save_node(key_file)

    @classmethod
    def add_standalone_node(cls) -> NoReturn:
        pass

    @classmethod
    def load_node(cls) -> NoReturn:
        """
        加载节点目录中的key文件

        :return:
        """
        key_dir = PathUtils.join(GflConf.home_dir, "key")
        cls.__load_node(PathUtils.join(key_dir, "key.json"))

    @classmethod
    def sign(cls, message: AnyStr) -> str:
        """

        :param message:
        :return:
        """
        if type(message) == str:
            message = message.encode("utf8")
        if type(message) != bytes:
            raise TypeError("message only support str or bytes.")
        encoded_message = encode_defunct(hexstr=message.hex())
        signed_message = w3.eth.account.sign_message(encoded_message, cls.priv_key)
        return signed_message.signature.hex()

    @classmethod
    def recover(cls, message: AnyStr, signature: str) -> str:
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

    @classmethod
    def encrypt(cls, plain: AnyStr, pub_key) -> bytes:
        """
        Encrypt with receiver's public key

        :param plain: data to encrypt
        :param pub_key: public key
        :return: encrypted data
        """
        if type(plain) == str:
            plain = plain.encode("utf8")
        if type(plain) != bytes:
            raise TypeError("message only support str or bytes.")
        cipher = ecies.encrypt(pub_key, plain)
        return cipher

    @classmethod
    def decrypt(cls, cipher: bytes) -> bytes:
        """
        Decrypt with private key

        :param cipher:
        :return:
        """
        if type(cipher) != bytes:
            raise TypeError("cipher only support bytes.")
        return ecies.decrypt(cls.priv_key, cipher)

    @classmethod
    def __new_node(cls):
        account = w3.eth.account.create()
        priv_key = keys.PrivateKey(account.key)
        pub_key = priv_key.public_key.to_hex()
        cls.__address = account.address[2:]
        cls.__pub_key = pub_key[2:]
        cls.__priv_key = priv_key.to_hex()[2:]

    @classmethod
    def __save_node(cls, path):
        d = {
            "address": cls.address,
            "pub_key": cls.pub_key,
            "priv_key": cls.priv_key
        }
        with open(path, "w") as f:
            f.write(json.dumps(d, indent=4))

    @classmethod
    def __load_node(cls, path):
        with open(path, "r") as f:
            keyjson = json.loads(f.read())
            cls.__address = keyjson["address"]
            cls.__pub_key = keyjson["pub_key"]
            cls.__priv_key = keyjson["priv_key"]
