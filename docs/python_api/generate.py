import inspect
import numbers
import os
import sys

import ctranslate2


def document_class(output_dir, class_path, base_path=None, children_paths=None):
    with open(os.path.join(output_dir, "%s.rst" % class_path), "w") as doc:
        doc.write("%s\n" % class_path)
        doc.write("=" * len(class_path))
        doc.write("\n\n")
        doc.write(".. autoclass:: %s\n" % class_path)
        doc.write("    :members:\n")
        doc.write("    :undoc-members:\n")
        doc.write("    :inherited-members:\n")
        if base_path:
            doc.write("\n    **Inherits from:** :class:`%s`\n" % base_path)
        if children_paths:
            doc.write("\n    **Extended by:**\n\n")
            for path in children_paths:
                doc.write("    - :class:`%s`\n" % path)


def document_function(output_dir, function_path):
    with open(os.path.join(output_dir, "%s.rst" % function_path), "w") as doc:
        doc.write("%s\n" % function_path)
        doc.write("=" * len(function_path))
        doc.write("\n\n")
        doc.write(".. autofunction:: %s\n" % function_path)


def module_is_public(module):
    return (
        module.__name__.startswith("ctranslate2")
        and hasattr(module, "__file__")
        and module.__file__.endswith("__init__.py")
    )


def get_module_map(module, module_path):
    """Map true modules to exported name"""
    if not module_is_public(module):
        return {}
    m = {}
    for symbol_name in dir(module):
        if symbol_name.startswith("_"):
            continue
        symbol = getattr(module, symbol_name)
        symbol_path = "%s.%s" % (module_path, symbol_name)
        m[symbol] = symbol_path
        if inspect.ismodule(symbol):
            m.update(get_module_map(symbol, symbol_path))
    return m


def get_first_public_parent(cls):
    base = cls.__bases__[0]
    while base.__name__.startswith("_"):  # Skip private parent classes.
        base = base.__bases__[0]
    if base is not object and base.__bases__[0] is tuple:  # For namedtuples.
        base = tuple
    return base


def annotate_classes(classes):
    annotations = []
    child_classes = {}
    for cls, path in classes:
        parent = get_first_public_parent(cls)
        if parent not in child_classes:
            child_classes[parent] = [cls]
        else:
            child_classes[parent].append(cls)
        annotations.append(dict(cls=cls, path=path, parent=parent))
    for annotation in annotations:
        annotation["children"] = child_classes.get(annotation["cls"])
    return annotations


def document_module(module, module_path, module_map, output_dir):
    if not module_is_public(module):
        return False
    submodules = []
    classes = []
    functions = []
    constants = []
    for symbol_name in dir(module):
        if symbol_name.startswith("_"):
            continue
        symbol = getattr(module, symbol_name)
        symbol_path = "%s.%s" % (module_path, symbol_name)
        if inspect.isclass(symbol):
            classes.append((symbol, symbol_path))
        elif inspect.isfunction(symbol) or inspect.ismethod(symbol) or inspect.isroutine(symbol):
            functions.append(symbol_path)
        elif inspect.ismodule(symbol):
            submodules.append((symbol_path, symbol))
        elif isinstance(symbol, (numbers.Number, str)):
            constants.append(symbol_path)

    with open(os.path.join(output_dir, "%s.rst" % module_path), "w") as doc:
        doc.write("%s module\n" % module_path)
        doc.write("=" * (len(module_path) + 7))
        doc.write("\n\n")
        doc.write(".. automodule:: %s\n\n" % module_path)

        if classes:
            doc.write("Classes\n")
            doc.write("-------\n\n")
            doc.write(".. toctree::\n\n")
            for class_info in annotate_classes(classes):
                base = class_info["parent"]
                base_path = module_map.get(
                    base, "%s.%s" % (base.__module__, base.__name__)
                )
                children_paths = class_info["children"]
                if children_paths:
                    children_paths = [
                        module_map.get(
                            child, "%s.%s" % (child.__module__, child.__name__)
                        )
                        for child in children_paths
                    ]
                class_path = class_info["path"]
                doc.write("   %s\n" % class_path)
                document_class(
                    output_dir,
                    class_path,
                    base_path=base_path,
                    children_paths=children_paths,
                )

        if functions:
            doc.write("\nFunctions\n")
            doc.write("---------\n\n")
            doc.write(".. toctree::\n\n")
            for function_path in functions:
                doc.write("   %s\n" % function_path)
                document_function(output_dir, function_path)

        if constants:
            doc.write("\nConstants\n")
            doc.write("---------\n\n")
            for constant_path in constants:
                doc.write("* %s\n" % constant_path)

        if submodules:
            submodules = list(
                filter(
                    lambda x: document_module(x[1], x[0], module_map, output_dir),
                    submodules,
                )
            )

            if submodules:
                doc.write("\nSubmodules\n")
                doc.write("----------\n\n")
                doc.write(".. toctree::\n\n")
                for module_path, module in submodules:
                    doc.write("   %s\n" % module_path)

        return True


output_dir = sys.argv[1]
os.makedirs(output_dir)
module_map = get_module_map(ctranslate2, "ctranslate2")
document_module(ctranslate2, "ctranslate2", module_map, output_dir)
