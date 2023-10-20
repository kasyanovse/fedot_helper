import sys
import inspect

from pathlib import Path


class SubRepository(dict):
    def __init__(self, folder: str):
        folder = Path(__file__).parent / folder
        files = set(x for x in folder.glob('*.py') if not (x.stem.startswith('_') or x.stem == folder.stem))

        # define classes
        classes = []
        for file in files:
            module = f"{folder.stem}.{file.stem}"
            __import__(module, level=0)
            for class_name, class_ in inspect.getmembers(sys.modules[module], inspect.isclass):
                if class_.__module__ == module:
                    classes.append((class_name, class_))

        classes = sorted(classes, key = lambda x: x[0])
        super().__init__(classes)

    def __setitem__(self, key, value):
        raise AttributeError('Cannot modify repository')


class Repository:
    estimator = SubRepository('estimator')
    ensemble = SubRepository('ensemble')
    task_transformer = SubRepository('task_transformer')

# TODO add tests for Repository
