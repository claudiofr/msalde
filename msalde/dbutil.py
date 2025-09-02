import math
from sqlalchemy import event


class StdDevExtension:
    def __init__(self):
        self.count = 0
        self.mean = 0
        self.M2 = 0

    def step(self, value):
        if value is None:
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    def finalize(self):
        if self.count < 2:
            return None
        return math.sqrt(self.M2 / (self.count - 1))


def register_extensions(dbapi_connection, connection_record):
    dbapi_connection.create_aggregate("stddev", 1, StdDevExtension)


class DbExtensionCreator:
    def __init__(self, engine):
        self._engine = engine
    
    def create_extensions(self):
        event.listen(self._engine, "connect", register_extensions)


