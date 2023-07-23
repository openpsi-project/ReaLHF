import getpass
import os
import tempfile


def get_user_tmp():
    tmp = tempfile.gettempdir()
    user = getpass.getuser()
    user_tmp = os.path.join(tmp, user)
    os.makedirs(user_tmp, exist_ok=True)
    return user_tmp


def get_random_tmp():
    return tempfile.mkdtemp()
