import os
from io import BytesIO
from pathlib import PurePath
from typing import NoReturn, Union
from zipfile import ZipFile, ZIP_DEFLATED


class ZipUtils(object):

    @classmethod
    def get_compress_data(cls, src_paths: Union[str, list], basename=None) -> bytes:
        """

        :param src_paths:
        :param basename:
        :return:
        """
        zip_file = BytesIO()
        cls.compress(src_paths, zip_file, basename)
        zip_file.seek(0)
        data = zip_file.read()
        zip_file.close()
        return data

    @classmethod
    def extract_data(cls, data: bytes, dst_path: str) -> NoReturn:
        """

        :param data:
        :param dst_path:
        :return:
        """
        zip_file = BytesIO()
        zip_file.write(data)
        zip_file.seek(0)
        cls.extract(zip_file, dst_path)
        zip_file.close()

    @classmethod
    def compress(cls, src_paths: Union[str, list], dst_zip_file, basename=None) -> NoReturn:
        """
        Compress the file that scr_paths point to into ZIP format and save it in dst_zip_file.

        :param src_paths: A compressed file or folder directory. Accepts multiple files or folders as a list.
        :param dst_zip_file: A file-like object or file path that stores compressed data
        :param basename: The root directory of the list of zip files.
        """
        if type(src_paths) not in [list, tuple, set]:
            src_paths = (src_paths, )
        if basename is None:
            basename = cls.__detect_basename(src_paths)
        if isinstance(dst_zip_file, str) and os.path.isdir(dst_zip_file):
            zip_filename = basename + ".zip"
            dst_zip_file = PurePath(dst_zip_file, zip_filename).as_posix()
        zip_file = ZipFile(dst_zip_file, "w", ZIP_DEFLATED)
        for p in src_paths:
            cls.__add_file(zip_file, basename, p)
        zip_file.close()

    @classmethod
    def extract(cls, src_zip_file, dst_path: str) -> NoReturn:
        """
        Unzip the zip file to the specified directory.

        :param src_zip_file: The file-like object or file path to be extracted
        :param dst_path: The directory that the zip file is unzipped to.
        """
        zip_file = ZipFile(src_zip_file, "r", ZIP_DEFLATED)
        zip_file.extractall(dst_path)
        zip_file.close()

    @classmethod
    def __add_file(cls, zip_file: ZipFile, basename, source):
        if not os.path.isdir(source):
            zip_file.write(source, basename)
            return
        for filename in os.listdir(source):
            new_source = PurePath(source, filename).as_posix()
            new_basename = PurePath(basename, filename).as_posix()
            if os.path.isdir(new_source):
                cls.__add_file(zip_file, new_basename, new_source)
            else:
                zip_file.write(new_source, new_basename)

    @classmethod
    def __detect_basename(cls, src_paths):
        if len(src_paths) == 1:
            return os.path.basename(src_paths[0])
        else:
            parent_dirname = os.path.dirname(src_paths[0])
            for p in src_paths[1:]:
                if parent_dirname != os.path.dirname(p):
                    return ""
            return os.path.basename(parent_dirname)
