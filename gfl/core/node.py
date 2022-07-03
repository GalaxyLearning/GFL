#  Copyright 2020 The GFL Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import base64
import json

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


def encode(bs: bytes, encoding: str):
    if encoding == "bytes":
        return bs
    elif encoding == "base64":
        return base64.b64encode(bs)
    elif encoding == "hex":
        return bs.hex().encode("ascii")
    else:
        raise ValueError(f"Unsupported encoding({encoding})")


def decode(bs, encoding: str) -> bytes:
    if encoding == "bytes":
        return bs
    elif encoding == "base64":
        return base64.b64decode(bs)
    elif encoding == "hex":
        return bytes.fromhex(bs.decode("ascii"))
    else:
        raise ValueError(f"Unsupported encoding({encoding})")


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

    def sign(self, message: bytes) -> str:
        """

        :param message:
        :return:
        """
        if type(message) != bytes:
            raise TypeError("message must be bytes.")
        encoded_message = encode_defunct(hexstr=message.hex())
        signed_message = w3.eth.account.sign_message(encoded_message, self.__priv_key)
        return signed_message.signature.hex()

    def recover(self, message: bytes, signature: str) -> str:
        """
        Get the address of the manager that signed the given message.

        :param message: the message that was signed
        :param signature: the signature of the message
        :return: the address of the manager
        """
        if self.__priv_key is None:
            raise ValueError(f"private key is None")
        if type(message) != bytes:
            raise TypeError("message must be bytes.")
        encoded_message = encode_defunct(message)
        return w3.eth.account.recover_message(encoded_message, signature=signature)

    def verify(self, message: bytes, signature: str, source_address: str) -> bool:
        """
        Verify whether the message is signed by source address

        :param message: the message that was signed
        :param signature: the signature of the message
        :param source_address: the message sent from
        :return: True or False
        """
        rec_addr = self.recover(message, signature)
        return rec_addr[2:].lower() == source_address.lower()

    def encrypt(self, plain: bytes, encoding="hex") -> bytes:
        """
        Encrypt with receiver's public key

        :param plain: data to encrypt
        :param encoding: the encoding type of encrypted data, only can be 'bytes', 'base64', or 'hex'
        :return: encrypted data
        """
        if type(plain) != bytes:
            raise TypeError("message must be bytes.")
        cipher = ecies.encrypt(self.__pub_key, plain)
        return encode(cipher, encoding)

    def decrypt(self, cipher: bytes, encoding="hex") -> bytes:
        """
        Decrypt with private key

        :param cipher: encrypted data
        :param encoding: the encoding type of encrypted data, only can be 'bytes', 'base64', or 'hex'
        :return:
        """
        if self.__priv_key is None:
            raise ValueError(f"private key is None")
        if type(cipher) != bytes:
            raise TypeError("cipher only support bytes.")
        cipher = decode(cipher, encoding)
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
