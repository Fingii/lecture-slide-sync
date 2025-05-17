import libcst as cst
import os


class DebugCleaner(cst.CSTTransformer):
    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        # Remove functions with "DEBUG ONLY" in docstring
        for stmt in updated_node.body.body:
            if isinstance(stmt, cst.SimpleStatementLine):
                for expr in stmt.body:
                    if isinstance(expr, cst.Expr) and isinstance(
                        expr.value, cst.SimpleString
                    ):
                        docstring = expr.value.value
                        if "DEBUG ONLY" in docstring:
                            return cst.RemoveFromParent()
        return updated_node

    def leave_If(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        # Remove "if DEBUG_MODE:" blocks
        if isinstance(updated_node.test, cst.Name) and updated_node.test.value == "DEBUG_MODE":
            return cst.RemoveFromParent()
        return updated_node

    def leave_Assign(
        self, original_node: cst.Assign, updated_node: cst.Assign
    ) -> cst.Assign:
        # Remove DEBUG_MODE assignments
        for target in updated_node.targets:
            if isinstance(target.target, cst.Name) and target.target.value == "DEBUG_MODE":
                return cst.RemoveFromParent()
        return updated_node

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom:
        # Remove "from X import pyautogui" statements
        for name in updated_node.names:
            if isinstance(name.name, cst.Name) and name.name.value == "pyautogui":
                return cst.RemoveFromParent()
        return updated_node

    def leave_Import(self, original_node: cst.Import, updated_node: cst.Import) -> cst.Import:
        # Remove "import pyautogui" statements
        for name in updated_node.names:
            if isinstance(name.name, cst.Name) and name.name.value == "pyautogui":
                return cst.RemoveFromParent()
        return updated_node


def clean_debug_code(file_path):
    with open(file_path, "r") as file:
        source_code = file.read()

    tree = cst.parse_module(source_code)
    transformer = DebugCleaner()
    modified_tree = tree.visit(transformer)

    directory = os.path.dirname(file_path)
    new_file_path = os.path.join(directory, "clean-version.py")

    with open(new_file_path, "w") as file:
        file.write(modified_tree.code)

    print(f"Cleaned version saved as: {new_file_path}")


# Example usage
clean_debug_code("slideDetection.py")