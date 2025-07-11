import pytest
from src.mother import QuantumMother

def test_spawn_prophets():
    mother = QuantumMother(0, 0)
    mother.spawn_prophets(3)
    assert len(mother.prophets) == 3
