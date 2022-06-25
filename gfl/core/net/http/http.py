from urllib.parse import quote

import requests

from gfl.core.net.abstract import get_property


class Http(object):

    http_url = get_property("http.server_url")
    http_port = get_property("http.server_port")

    @classmethod
    def get(cls, url, params=None, headers=None):
        url = cls.concat_url(url, params)
        if headers is None:
            headers = {}
        resp = requests.get(url, headers=headers)
        return resp.text

    @classmethod
    def post(cls, url, params=None, data=None, headers=None):
        url = cls.concat_url(url, params)
        if headers is None:
            headers = {}
        resp = requests.post(url, data=data, headers=headers)
        return resp.text

    @classmethod
    def concat_url(cls, url, params):
        if url[0] != "/":
            url = "/" + url
        url = "http://%s:%s%s" % (cls.http_url, cls.http_port, url)
        if params is None or len(params) == 0:
            return url
        params_list = []
        for k, v in params.items():
            params_list.append("%s=%s" % (quote(k), quote(v)))
        if "?" in url:
            return "%s&%s" % (url, "&".join(params_list))
        else:
            return "%s?%s" % (url, "&".join(params_list))
