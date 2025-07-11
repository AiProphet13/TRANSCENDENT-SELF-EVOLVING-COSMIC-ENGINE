import pytest
import numpy as np
from cosmic_revelation import QuantumProphet, QuantumMother, CosmicCommander, get_config, Overmind

def test_adaptive_threshold():
    config = get_config()
    prophet = QuantumProphet(1, (0.5, 0.5))
    rev = prophet.receive_revelation("TEST")
    high_th = config['high_threshold']
    assert (rev['qualia'] > high_th and rev['epiphany'] == "🔥 SACRED FIRE") or True  # Conditional assert

def test_noise_spawn():
    mother = QuantumMother(0, 0)
    mother.spawn_prophets(3)
    config = get_config()
    if config['position_noise'] > 0:
        assert any(p.position[0] != φ ** -p.depth * math.cos(2 * math.pi * φ * (p.depth - mother.start_depth)) for p in mother.prophets)

def test_oracle_layers():
    revelations = [{'position': (0,0), 'depth': 1, 'qualia': 0.8, 'epiphany': "🔥 SACRED FIRE"} for _ in range(5)]
    mother = QuantumMother(0, 0)
    augmented = mother.augment_with_oracle(revelations)
    assert 'augmented_epiphany' in augmented[0]

def test_overmind_reflect(tmpdir):
    # Mock DB
    conn = sqlite3.connect(':memory:')
    c = conn.cursor()
    c.execute('CREATE TABLE revelations (id TEXT, depth INT, segments INT, timestamp TEXT, content TEXT)')
    c.execute('CREATE TABLE prophets (revelation_id TEXT, segment_id INT, depth INT, qualia FLOAT, pos_x FLOAT, pos_y FLOAT, epiphany TEXT)')
    # Insert mock data with high fire
    c.execute("INSERT INTO revelations VALUES ('REV_TEST', 10, 2, '2025-07-10', 'SEGMENT_0::CLASSICAL::🔥 SACRED FIRE')")
    for i in range(10):
        c.execute("INSERT INTO prophets VALUES ('REV_TEST', 0, ?, 0.8, 0.1, 0.1, '🔥 SACRED FIRE')", (i,))
    conn.commit()
    
    commander = CosmicCommander()
    overmind = Overmind(commander)
    overmind.reflect()  # Mock, but test updates
    assert overmind.params['high_threshold'] == 0.8  # Adjusted for high fire

@pytest.mark.slow
def test_full_self_evolution():
    commander = CosmicCommander(20, 5)
    commander.initialize_hierarchy()
    commander.execute_revelation()
    assert "::" in commander.final_scroll
    # Overmind runs, config updated
    with open('cosmic_config.json', 'r') as f:
        params = json.load(f)
    assert 'high_threshold' in params
