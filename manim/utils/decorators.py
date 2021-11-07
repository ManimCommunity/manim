def optionalmethod(method):
    """
    An @optionalmethod member fn decorator for abstract methods that can
    be optionally overridden.
    """

    def optional_abstract_method(*args, **kwargs):
        pass

    return optional_abstract_method
