from pithon.evaluator.envframe import EnvFrame
from pithon.evaluator.primitive import check_type, get_primitive_dict
from pithon.syntax import (
    PiAssignment, PiBinaryOperation, PiNumber, PiBool, PiStatement, PiProgram, PiSubscript, PiVariable,
    PiIfThenElse, PiNot, PiAnd, PiOr, PiWhile, PiNone, PiList, PiTuple, PiString,
    PiFunctionDef, PiFunctionCall, PiFor, PiBreak, PiContinue, PiIn, PiReturn,
    PiClassDef, PiAttribute, PiAttributeAssignment
)
from pithon.evaluator.envvalue import (
    EnvValue, VFunctionClosure, VList, VNone, VTuple, VNumber, VBool, VString,
    VClassDef, VObject, VMethodClosure
)


def initial_env() -> EnvFrame:
    """Crée et retourne l'environnement initial avec les primitives."""
    env = EnvFrame()
    env.vars.update(get_primitive_dict())
    return env


def lookup(env: EnvFrame, name: str) -> EnvValue:
    """Recherche une variable dans l'environnement."""
    return env.lookup(name)


def insert(env: EnvFrame, name: str, value: EnvValue) -> None:
    """Insère une variable dans l'environnement."""
    env.insert(name, value)


def evaluate(node: PiProgram, env: EnvFrame) -> EnvValue:
    """Évalue un programme ou une liste d'instructions."""
    if isinstance(node, list):
        last_value = VNone(value=None)
        for stmt in node:
            last_value = evaluate_stmt(stmt, env)
        return last_value
    elif isinstance(node, PiStatement):
        return evaluate_stmt(node, env)
    else:
        raise TypeError(f"Type de nœud non supporté : {type(node)}")


def evaluate_stmt(node: PiStatement, env: EnvFrame) -> EnvValue:
    """Évalue une instruction ou expression Pithon."""

    if isinstance(node, PiNumber):
        return VNumber(node.value)

    elif isinstance(node, PiBool):
        return VBool(node.value)

    elif isinstance(node, PiNone):
        return VNone(node.value)

    elif isinstance(node, PiString):
        return VString(node.value)

    elif isinstance(node, PiList):
        elements = [evaluate_stmt(e, env) for e in node.elements]
        return VList(elements)

    elif isinstance(node, PiTuple):
        elements = tuple(evaluate_stmt(e, env) for e in node.elements)
        return VTuple(elements)

    elif isinstance(node, PiVariable):
        return lookup(env, node.name)

    elif isinstance(node, PiBinaryOperation):
        # Traite l'opération binaire comme un appel de fonction
        fct_call = PiFunctionCall(
            function=PiVariable(name=node.operator),
            args=[node.left, node.right]
        )
        return evaluate_stmt(fct_call, env)

    elif isinstance(node, PiAssignment):
        value = evaluate_stmt(node.value, env)
        insert(env, node.name, value)
        return value

    elif isinstance(node, PiAttributeAssignment):
        obj = evaluate_stmt(node.object, env)
        if not isinstance(obj, VObject):
            raise TypeError(f"Impossible d'assigner un attribut à un objet de type {type(obj).__name__}")
        value = evaluate_stmt(node.value, env)
        obj.attributes[node.attr] = value
        return value

    elif isinstance(node, PiAttribute):
        obj = evaluate_stmt(node.object, env)
        if not isinstance(obj, VObject):
            raise TypeError(f"Impossible d'accéder à un attribut d'un objet de type {type(obj).__name__}")

        # Vérifier d'abord les attributs d'instance
        if node.attr in obj.attributes:
            return obj.attributes[node.attr]

        # Puis vérifier les méthodes de la classe
        if node.attr in obj.class_def.methods:
            method = obj.class_def.methods[node.attr]
            return VMethodClosure(function=method, instance=obj)

        raise AttributeError(f"L'objet {obj.class_def.name} n'a pas d'attribut '{node.attr}'")

    elif isinstance(node, PiIfThenElse):
        cond = evaluate_stmt(node.condition, env)
        cond = check_type(cond, VBool)
        branch = node.then_branch if cond.value else node.else_branch
        last_value = evaluate(branch, env)
        return last_value

    elif isinstance(node, PiNot):
        operand = evaluate_stmt(node.operand, env)
        # Vérifie le type pour l'opérateur 'not'
        _check_valid_piandor_type(operand)
        return VBool(not operand.value)  # type: ignore

    elif isinstance(node, PiAnd):
        left = evaluate_stmt(node.left, env)
        _check_valid_piandor_type(left)
        if not left.value:  # type: ignore
            return left
        right = evaluate_stmt(node.right, env)
        _check_valid_piandor_type(right)
        return right

    elif isinstance(node, PiOr):
        left = evaluate_stmt(node.left, env)
        _check_valid_piandor_type(left)
        if left.value:  # type: ignore
            return left
        right = evaluate_stmt(node.right, env)
        _check_valid_piandor_type(right)
        return right

    elif isinstance(node, PiWhile):
        return _evaluate_while(node, env)

    elif isinstance(node, PiFunctionDef):
        closure = VFunctionClosure(node, env)
        insert(env, node.name, closure)
        return VNone(value=None)

    elif isinstance(node, PiClassDef):
        # Créer un environnement pour la classe
        class_env = EnvFrame(parent=env)

        # Évaluer les méthodes dans l'environnement de la classe
        methods = {}
        for method in node.methods:
            method_closure = VFunctionClosure(method, class_env)
            methods[method.name] = method_closure

        # Créer la définition de classe
        class_def = VClassDef(name=node.name, methods=methods)

        # Insérer la classe dans l'environnement
        insert(env, node.name, class_def)
        return VNone(value=None)

    elif isinstance(node, PiReturn):
        value = evaluate_stmt(node.value, env)
        raise ReturnException(value)

    elif isinstance(node, PiFunctionCall):
        return _evaluate_function_call(node, env)

    elif isinstance(node, PiFor):
        return _evaluate_for(node, env)

    elif isinstance(node, PiBreak):
        raise BreakException()

    elif isinstance(node, PiContinue):
        raise ContinueException()

    elif isinstance(node, PiIn):
        return _evaluate_in(node, env)

    elif isinstance(node, PiSubscript):
        return _evaluate_subscript(node, env)

    else:
        raise TypeError(f"Type de nœud non supporté : {type(node)}")


def _check_valid_piandor_type(obj):
    """Vérifie que le type est valide pour 'and'/'or'."""
    if not isinstance(obj, VBool | VNumber | VString | VNone | VList | VTuple):
        raise TypeError(f"Type non supporté pour l'opérateur 'and': {type(obj).__name__}")


def _evaluate_while(node: PiWhile, env: EnvFrame) -> EnvValue:
    """Évalue une boucle while."""
    last_value = VNone(value=None)
    while True:
        cond = evaluate_stmt(node.condition, env)
        cond = check_type(cond, VBool)
        if not cond.value:
            break
        try:
            last_value = evaluate(node.body, env)
        except BreakException:
            break
        except ContinueException:
            continue
    return last_value


def _evaluate_for(node: PiFor, env: EnvFrame) -> EnvValue:
    """Évalue une boucle for."""
    iterable_val = evaluate_stmt(node.iterable, env)
    if not isinstance(iterable_val, (VList, VTuple)):
        raise TypeError("La boucle for attend une liste ou un tuple.")
    last_value = VNone(value=None)
    iterable = iterable_val.value
    for item in iterable:
        env.insert(node.var, item)  # Pas de nouvel environnement pour la variable de boucle
        try:
            last_value = evaluate(node.body, env)
        except BreakException:
            break
        except ContinueException:
            continue
    return last_value


def _evaluate_subscript(node: PiSubscript, env: EnvFrame) -> EnvValue:
    """Évalue une opération d'indexation (subscript)."""
    collection = evaluate_stmt(node.collection, env)
    index = evaluate_stmt(node.index, env)
    # Indexation pour liste, tuple ou chaîne
    if isinstance(collection, VList):
        idx = check_type(index, VNumber)
        try:
            return collection.value[int(idx.value)]
        except IndexError:
            raise IndexError("Index de liste hors limites")
    elif isinstance(collection, VTuple):
        idx = check_type(index, VNumber)
        try:
            return collection.value[int(idx.value)]
        except IndexError:
            raise IndexError("Index de tuple hors limites")
    elif isinstance(collection, VString):
        idx = check_type(index, VNumber)
        try:
            return VString(collection.value[int(idx.value)])
        except IndexError:
            raise IndexError("Index de chaîne hors limites")
    else:
        raise TypeError("L'indexation n'est supportée que pour les listes, tuples et chaînes.")


def _evaluate_in(node: PiIn, env: EnvFrame) -> EnvValue:
    """Évalue l'opérateur 'in'."""
    container = evaluate_stmt(node.container, env)
    element = evaluate_stmt(node.element, env)
    if isinstance(container, (VList, VTuple)):
        return VBool(element in container.value)
    elif isinstance(container, VString):
        if isinstance(element, VString):
            return VBool(element.value in container.value)
        else:
            return VBool(False)
    else:
        raise TypeError("'in' n'est supporté que pour les listes et chaînes.")


def _evaluate_function_call(node: PiFunctionCall, env: EnvFrame) -> EnvValue:
    """Évalue un appel de fonction (primitive, définie par l'utilisateur, ou constructeur de classe)."""
    func_val = evaluate_stmt(node.function, env)
    args = [evaluate_stmt(arg, env) for arg in node.args]
    # Fonction primitive
    if callable(func_val):
        return func_val(args)

    # Constructeur de classe
    if isinstance(func_val, VClassDef):
        # Créer une nouvelle instance
        instance = VObject(class_def=func_val, attributes={})

        # Appeler __init__ si elle existe
        if "__init__" in func_val.methods:
            init_method = func_val.methods["__init__"]
            # Créer l'environnement d'appel pour __init__
            call_env = EnvFrame(parent=init_method.closure_env)

            # Le premier argument est 'self'
            if len(init_method.funcdef.arg_names) > 0:
                call_env.insert(init_method.funcdef.arg_names[0], instance)

            # Ajouter les autres arguments
            for i, arg_name in enumerate(init_method.funcdef.arg_names[1:], 1):
                if i-1 < len(args):
                    call_env.insert(arg_name, args[i-1])
                else:
                    raise TypeError(f"Argument manquant pour __init__: {arg_name}")

            # Exécuter __init__
            try:
                for stmt in init_method.funcdef.body:
                    evaluate_stmt(stmt, call_env)
            except ReturnException:
                pass  # __init__ peut retourner quelque chose, on l'ignore

        return instance

    # Méthode liée
    if isinstance(func_val, VMethodClosure):
        funcdef = func_val.function.funcdef
        closure_env = func_val.function.closure_env
        instance = func_val.instance

        # Créer l'environnement d'appel
        call_env = EnvFrame(parent=closure_env)

        # Le premier argument est 'self'
        if len(funcdef.arg_names) > 0:
            call_env.insert(funcdef.arg_names[0], instance)

        # Ajouter les autres arguments
        for i, arg_name in enumerate(funcdef.arg_names[1:], 1):
            if i-1 < len(args):
                call_env.insert(arg_name, args[i-1])
            else:
                raise TypeError(f"Argument manquant pour la méthode {funcdef.name}: {arg_name}")

        # Gérer varargs si nécessaire
        if funcdef.vararg:
            varargs = VList(args[len(funcdef.arg_names)-1:])
            call_env.insert(funcdef.vararg, varargs)
        elif len(args) > len(funcdef.arg_names) - 1:
            raise TypeError("Trop d'arguments pour la méthode.")

        # Exécuter la méthode
        result = VNone(value=None)
        try:
            for stmt in funcdef.body:
                result = evaluate_stmt(stmt, call_env)
        except ReturnException as ret:
            return ret.value

        return result

    # Fonction utilisateur
    if isinstance(func_val, VFunctionClosure):
        funcdef = func_val.funcdef
        closure_env = func_val.closure_env
        call_env = EnvFrame(parent=closure_env)

        # Ajouter les arguments
        for i, arg_name in enumerate(funcdef.arg_names):
            if i < len(args):
                call_env.insert(arg_name, args[i])
            else:
                raise TypeError(f"Argument manquant pour la fonction {funcdef.name}: {arg_name}")

        # Gérer varargs
        if funcdef.vararg:
            varargs = VList(args[len(funcdef.arg_names):])
            call_env.insert(funcdef.vararg, varargs)
        elif len(args) > len(funcdef.arg_names):
            raise TypeError("Trop d'arguments pour la fonction.")

        # Exécuter la fonction
        result = VNone(value=None)
        try:
            for stmt in funcdef.body:
                result = evaluate_stmt(stmt, call_env)
        except ReturnException as ret:
            return ret.value

        return result

    raise TypeError(f"Tentative d'appel d'un objet non-fonction de type {type(func_val).__name__}")


class ReturnException(Exception):
    """Exception pour retourner une valeur depuis une fonction."""
    def __init__(self, value):
        self.value = value


class BreakException(Exception):
    """Exception pour sortir d'une boucle (break)."""
    pass


class ContinueException(Exception):
    """Exception pour passer à l'itération suivante (continue)."""
    pass
