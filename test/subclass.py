class Waitable:
    def __init__(self):
        self._finished = False

    def is_completed(self):
        return self._finished

    def wait(self):
        if self._finished:
            return
        self._wait_task()
        self._finished = True

    def _wait_task():
        # Implement this method in the subclass.
        ...


class DummyHandle(Waitable):
    def _wait_task(self):
        pass


class FunctionHandle(Waitable):
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self._function = function
        self._args = args
        self._kwargs = kwargs

    def _wait_task(self):
        self._function(*self._args, **self._kwargs)


a = FunctionHandle(print, 'hello')
a.wait()
a.wait()

b = DummyHandle()
b.wait()
b.wait()
