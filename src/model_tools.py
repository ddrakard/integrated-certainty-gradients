"""
    Tools for working with Keras models.
"""

import os

import keras


def ensure_directory_exists(path: str) -> None:
    """
        Make sure the path is a directory, create it if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    elif not os.path.isdir(path):
        raise Exception('Path is not a directory: ' + path)


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
    elif os.path.exists(base_path + index_separator + '0'):
        last_path = base_path + index_separator + '0'
    elif os.path.exists(base_path + index_separator + '1'):
        last_path = base_path + index_separator + '1'
    else:
        raise FileNotFoundError(
            'No initial path similar to the base path: ' + str(base_path)
            + ' was found.')
    index = 2
    while True:
        candidate_path = base_path + index_separator + str(index) + extension
        if os.path.exists(candidate_path):
            last_path = candidate_path
            index += 1
        else:
            return last_path


def safe_save_model(model: keras.Model, base_path: str) -> None:
    """
        Save the model auto-creating a directory so it will not overwrite or
        be blocked by an existing model.

        :param model: The model to save.
        :param base_path: The directory in which to save the model.
    """
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
    """
        Load the latest model saved by safe_save_model in a directory.
    """
    return keras.models.load_model(latest_version(base_path, ''))
