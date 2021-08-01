import os
import sys
import sqlite3


class WorkDirContext(object):
    """
    Temporarily modify the work directory
    """

    def __init__(self, workdir):
        super(WorkDirContext, self).__init__()
        self.workdir = workdir
        self.pre_workdir = os.getcwd()
        self.std_out = sys.stdout
        self.std_err = sys.stderr
        self.std_out_filename = "console_out"
        self.std_err_filename = "console_err"
        self.std_out_file = None
        self.std_err_file = None

    def __enter__(self):
        self.pre_workdir = os.getcwd()
        os.chdir(self.workdir)
        self.std_out_file = open(self.std_out_filename, "a")
        self.std_err_file = open(self.std_out_filename, "a")
        sys.stdout = self.std_out_file
        sys.stderr = self.std_err_file

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.pre_workdir)
        self.std_out_file.close()
        self.std_err_file.close()
        sys.stdout = self.std_out
        sys.stderr = self.std_err


class SqliteContext(object):

    def __init__(self, path):
        super(SqliteContext, self).__init__()
        self.path = path

    def __enter__(self):
        self.conn = sqlite3.connect(self.path)
        self.cursor = self.conn.cursor()
        return self.conn, self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cursor.close()
        self.conn.commit()
        self.conn.close()
