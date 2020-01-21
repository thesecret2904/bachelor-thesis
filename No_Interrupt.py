class NoInterrupt:
    def __init__(self, func):
        self.func = func

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.func(*self.args, **self.kwargs)

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        with self:
            pass
