import importlib
import inspect
import os
import shutil
from pathlib import PurePath


class ModuleUtils(object):

    @classmethod
    def get_module_path(cls, module):
        """

        :param module:
        :return:
        """
        module_path = inspect.getsourcefile(module)
        if module_path.endswith("__init__.py"):
            return PurePath(os.path.dirname(module_path)).as_posix()
        else:
            return PurePath(module_path).as_posix()

    @classmethod
    def migrate_module(cls, src_module_path: str, target_module_name, target_dir):
        """

        :param src_module_path:
        :param target_module_name:
        :param target_dir:
        :return:
        """
        if src_module_path.endswith("__init__.py"):
            shutil.copytree(os.path.dirname(src_module_path), PurePath(target_dir, target_module_name).as_posix())
        elif os.path.isdir(src_module_path):
            shutil.copytree(src_module_path, PurePath(target_dir, target_module_name).as_posix())
        else:
            shutil.copy(src_module_path, PurePath(target_dir, target_module_name + ".py").as_posix())

    @classmethod
    def submit_module(cls, module, target_module_name, target_dir):
        """
        copy a module to the target directory

        :param module: source module
        :param target_module_name: target module name
        :param target_dir: target directory
        """
        cls.verify_module_api(module)
        module_path = inspect.getsourcefile(module)
        cls.migrate_module(module_path, target_module_name, target_dir)

    @classmethod
    def import_module(cls, path, name):
        """
        Import the module dynamically

        :param path: module path
        :param name: module name
        :return: module
        """
        if name is not None:
            module_name = path.replace("/", ".") + "." + name
        else:
            module_name = path.replace("/", ".")
        return importlib.import_module(module_name)

    @classmethod
    def verify_module_api(cls, module):
        pass

    @classmethod
    def exists_module(cls, module):
        try:
            exec("import " + module)
            return True
        except:
            return False

    @classmethod
    def get_name(cls, module, obj):
        if module is None or obj is None:
            raise ValueError("")
        for k, v in module.__dict__.items():
            if v == obj:
                return k
        return None
