"""
    Run the pylint static code quality checker on this project.
"""
import os
import sys
from pathlib import Path
import pylint.lint


def lint_files(files_list):
    """
        Run pylint on the chosen files using setting appropriate to this
        project.
    """
    disabled_messages = [
        # This codebase was not intended to be used in production, and "to do"
        # comments are not considered malpractice.
        'fixme',
        # TODO: Refactor areas of similar code.
        'duplicate-code',
        # Including the argument keyword can made code clearer, even if the
        # argument is in positional-argument position already.
        'unexpected-keyword-arg',
        # Pylint does not seem to correctly recognise tensorflow optional
        # parameters.
        'no-value-for-parameter',
        # Else clauses can make the code easier to follow even if they are
        # not strictly necessary.
        'no-else-return',
        # This bloats the code in trivial cases. Could consider making this
        # more strict since it can be useful.
        'dangerous-default-value',
        # Pylint appears not to recognise tensorflow.python module.
        'no-name-in-module',
        # Iterators do not need public method definitions. This is true
        # generally for abstract implementations.
        'too-few-public-methods',
        # This appears to have a false positive.
        'invalid-unary-operand-type',
        # Appears to be a false positive for
        # tensorflow.python.ops.summary_ops_v2.ResourceSummaryWriter
        'not-context-manager',
        # unittest test methods may not reference the instance but should still
        # be methods of a test class.
        'no-self-use',
        'too-many-arguments',
        'too-many-statements',
        'too-many-branches',
        'too-many-locals',
        'too-many-public-methods',
    ]
    disabled_messages_argument = '--disable=' + ','.join(disabled_messages)
    good_variable_names = [
        '_',  # Python standard name
        'x',  # Standard machine learning terminology
        'y',  # Standard machine learning terminology
    ]
    good_variable_names_argument = '--good-names=' + ','.join(
        good_variable_names)
    pylint.lint.Run(
        [disabled_messages_argument, good_variable_names_argument]
        + files_list)


if __name__ == '__main__':
    SOURCE_DIRECTORY = Path(__file__).resolve().parent
    sys.path.insert(0, str(SOURCE_DIRECTORY))
    os.chdir(SOURCE_DIRECTORY.parent)
    FILES = [str(path) for path in Path("src").rglob("*.[pP][yY]")]
    lint_files(FILES)
