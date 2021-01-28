import os

import keras


def ensure_directory_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def unique_path(
        base_path: str, extension: str, qualify_first: bool = False,
        index_separator: str = '_') -> str:
    """
        Create a unique filename from a base filename. This assists when saving
        multiple file versions and prevents overwriting.

        :param base_path: The path and prefix of the filename to create. If it
            already exists a unique postfix will be added.
        :param extension: The desired filename extension, without a leading
            dot.
        :param qualify_first: If no file exists, if True the filename will be
            numbered "1", if False will have no number.
        :param index_separator: A character to place between the filename and
            any appended number.
        :return: An unoccupied filename.
    """
    if extension:
        extension = '.' + extension
    index = 1
    while True:
        if index == 1 and not qualify_first:
            path = base_path + extension
        else:
            path = base_path + index_separator + str(index) + extension
        if os.path.exists(path):
            index += 1
        else:
            return path


def latest_version(
        base_path: str, extension: str, index_separator: str = '_') -> str:
    """
        Get the most recent filename created with the unique_path function.

        :param base_path: The base path from which the versioned filenames will
            have been created. See unique_path.
        :param extension: The filename extension of the versioned filenames.
        :param index_separator: The character passed for index_separator to
            unique_path when the file was created.
        :return: The latest filename matching the base_path, extension, and
            index_separator.
    """
    if extension:
        extension = '.' + extension
    last_path = None
    if os.path.exists(base_path):
        last_path = base_path
    if os.path.exists(base_path + index_separator + '0'):
        last_path = base_path + index_separator + '0'
    if os.path.exists(base_path + index_separator + '1'):
        last_path = base_path + index_separator + '1'
    index = 2
    while True:
        candidate_path = base_path + index_separator + str(index) + extension
        if os.path.exists(candidate_path):
            last_path = candidate_path
            index += 1
        else:
            return last_path


def safe_save_model(model: keras.Model, base_path: str) -> None:
    keras.models.save_model(
        model,
        unique_path(base_path, ''),
        overwrite=False,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )


def load_latest_model(base_path: str) -> keras.Model:
    return keras.models.load_model(latest_version(base_path, ''))
