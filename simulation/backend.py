# cf. https://qiskit.org/documentation/apidoc/transpiler.html


import math, os
import qiskit
import qiskit.providers.fake_provider
from qiskit import QuantumCircuit


class MyFakeBackend(qiskit.providers.fake_provider.fake_backend.FakeBackendV2):
    dirname = os.path.dirname(__file__)
    conf_filename = 'data/conf_fake_backend.json'
    props_filename = 'data/props_fake_backend.json'
    defs_filename = 'data/defs_fake_backend.json'
    backend_name = 'my_fake_backend'
