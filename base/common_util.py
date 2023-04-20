# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/6/10

import os
import glob
import codecs
import json
from natsort import natsorted
from base.driver import logger, PROJECT_ROOT_PATH


def get_file_list(folder_path: str, p_postfix: list = None, sub_dir: bool = True) -> list:
    """
    获取所给文件目录里的指定后缀的文件,读取文件列表目前使用的是 os.walk 和 os.listdir ，这两个目前比 pathlib 快很多
    :param folder_path: 文件夹名称
    :param p_postfix: 文件后缀,如果为 [.*]将返回全部文件
    :param sub_dir: 是否搜索子文件夹
    :return: 获取到的指定类型的文件列表
    """
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)
    if p_postfix is None:
        p_postfix = ['.jpg']
    if isinstance(p_postfix, str):
        p_postfix = [p_postfix]
    logger.info("begin to get files from:{}".format(folder_path))
    file_list = [
        x for x in glob.glob(folder_path + '/**/*.*', recursive=True)
        if os.path.splitext(x)[-1].lower() in p_postfix or '*'.lower() in p_postfix
    ]
    logger.info("success to get files from:{}".format(folder_path))

    return natsorted(file_list)


def search_file_from_dir(file_dir, file_path_pref_list, base_pref="", exts=["pdf", "PDF"], save_pref=True):
    assert isinstance(file_path_pref_list, list)
    base_dir_name = os.path.basename(file_dir)
    logger.info("searching exts:{} from:{}".format(exts, file_dir))
    if len(base_pref.strip()) == 0:
        current_base_pref = ""
    else:
        current_base_pref = "{}_{}".format(base_pref, base_dir_name)
    for file_name in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file_name)
        if os.path.isdir(file_path):
            search_file_from_dir(file_path, file_path_pref_list, current_base_pref, exts, save_pref)
        else:
            file_res = file_name.rsplit(".", 1)
            if len(file_res) == 2:
                file_pref, file_ext = file_res
                if file_ext.strip().lower() in exts:
                    if save_pref:
                        if len(current_base_pref.strip()) > 0:
                            current_file_pref = "{}_{}".format(current_base_pref, file_pref)
                        else:
                            current_file_pref = "{}".format(file_pref)
                        file_path_pref_list.append((file_path, current_file_pref))
                    else:
                        file_path_pref_list.append(file_path)


def get_absolute_file_path(file_path):
    if file_path.startswith("/"):
        return file_path
    else:
        return os.path.join(PROJECT_ROOT_PATH, file_path)


def get_file_path_list(path, ext=None):
    if not path.startswith('/'):
        path = os.path.join(PROJECT_ROOT_PATH, path)
    # print(path)
    assert os.path.exists(path), 'path not exist {}'.format(path)
    assert ext is not None, 'ext is None'
    if os.path.isfile(path):
        return [path]
    file_path_list = []
    for root, _, files in os.walk(path):
        for file in files:
            try:
                if file.rsplit('.')[-1].lower() in ext:
                    file_path_list.append(os.path.join(root, file))
            except Exception as e:
                pass
    return file_path_list


# load json data
def load_json(data):
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        file_path_list = data
    elif data.endswith('.json'):
        file_path_list = [data]
    else:
        file_path_list = get_file_path_list(data, '.json')
    json_data_list = list()
    for file_path in file_path_list:
        with codecs.open(file_path, "r", "utf-8") as fr:
            json_data = json.loads(fr.read())
            json_data_list.append(json_data)
    return json_data_list


def save_params(save_dir, save_json, yml_name='config.yaml'):
    import yaml
    with open(os.path.join(save_dir, yml_name), 'w', encoding='utf-8') as f:
        yaml.dump(save_json, f, default_flow_style=False, encoding='utf-8', allow_unicode=True)


def read_config(config_file):
    import anyconfig
    if os.path.exists(config_file):
        with open(config_file, "rb") as fr:
            config = anyconfig.load(fr)
        if 'base' in config:
            base_config_path = config['base']
            if not base_config_path.startswith('/'):
                base_config_path = os.path.join(PROJECT_ROOT_PATH, base_config_path)
        elif os.path.basename(config_file) == 'base.yaml':
            return config
        else:
            base_config_path = os.path.join(os.path.dirname(config_file), "base.yaml")
        base_config = read_config(base_config_path)
        merged_config = base_config.copy()
        merge_config(config, merged_config)
        return merged_config
    else:
        return {}


def merge_config(config, base_config):
    for key, _ in config.items():
        if isinstance(config[key], dict) and key not in base_config:
            base_config[key] = config[key]
        elif isinstance(config[key], dict):
            merge_config(config[key], base_config[key])
        else:
            if key in base_config:
                base_config[key] = config[key]
            else:
                base_config.update({key: config[key]})


def init_experiment_config(config_file, experiment_name):
    if not config_file.startswith("/"):
        config_file = get_absolute_file_path(config_file)
    input_config = read_config(config_file)
    experiment_base_config = read_config(os.path.join(PROJECT_ROOT_PATH, 'config', experiment_name.lower(),
                                                      'base.yaml'))
    merged_config = experiment_base_config.copy()
    merge_config(input_config, merged_config)

    base_config = read_config(os.path.join(PROJECT_ROOT_PATH, 'config',
                                                      'base.yaml'))
    final_merged_config = base_config.copy()
    merge_config(merged_config, final_merged_config)
    return final_merged_config
