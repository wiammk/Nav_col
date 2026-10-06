import math

import numpy as np
import pandas as pd
import pytest

from data import ifc_parser
from data.graph_builder import validate_spatial_quality


def _node(node_id, x, y, z=0.0, floor="Floor_0", floor_elev=0.0):
    return {
        "id": node_id,
        "x": x,
        "y": y,
        "z": z,
        "floor": floor,
        "floor_elev": floor_elev,
    }


def test_complete_placement_matrix_keeps_rotation_and_scales_translation(monkeypatch):
    source = np.array(
        [
            [0.0, -1.0, 0.0, 1000.0],
            [1.0, 0.0, 0.0, 2000.0],
            [0.0, 0.0, 1.0, 3000.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    monkeypatch.setattr(ifc_parser, "get_local_placement", lambda placement: source)

    result = ifc_parser.get_local_placement_matrix(object(), unit_scale=0.001)

    assert np.allclose(result[:3, :3], source[:3, :3])
    assert np.allclose(result[:3, 3], [1.0, 2.0, 3.0])


def test_parser_rejects_kict_like_degenerate_coordinates():
    nodes = {str(i): _node(i, 0.0, 0.0, float(i)) for i in range(20)}
    edges = [
        {"from": i, "to": i + 1, "weight": 0.0}
        for i in range(19)
    ]

    with pytest.raises(ValueError, match="dégénéré"):
        ifc_parser.validate_spatial_data(nodes, edges)


def test_graph_builder_rejects_zero_distance_edges():
    nodes = pd.DataFrame(
        [_node(i, float(i % 5), float(i // 5)) for i in range(20)]
    )
    edges = pd.DataFrame(
        [
            {"from": i, "to": (i + 1) % 20, "weight": 1.0, "type": "door"}
            for i in range(20)
        ]
    )
    nodes.loc[1, ["x", "y", "z"]] = nodes.loc[0, ["x", "y", "z"]]
    for i in range(1, 10):
        edges.loc[i, "from"] = 0
        edges.loc[i, "to"] = 1

    with pytest.raises(ValueError, match="Graphe refusé"):
        validate_spatial_quality(nodes, edges)


def test_valid_spatial_graph_passes():
    nodes = {str(i): _node(i, float(i % 5), float(i // 5)) for i in range(20)}
    edges = [
        {
            "from": i,
            "to": i + 1,
            "weight": math.sqrt(1.0 if i % 5 != 4 else 17.0),
        }
        for i in range(19)
    ]

    report = ifc_parser.validate_spatial_data(nodes, edges)

    assert report["unique_xy_ratio"] == 1.0
    assert report["zero_distance_ratio"] == 0.0


def test_exclude_floor_removes_requested_storey_and_reindexes_nodes():
    nodes = {
        "a": _node(4, 0.0, 0.0, floor="First Floor", floor_elev=0.0),
        "b": _node(8, 1.0, 0.0, floor="Second Floor", floor_elev=4.5),
        "c": _node(12, 2.0, 0.0, floor="Roof - Main", floor_elev=9.0),
    }

    filtered = ifc_parser.exclude_floors(nodes, ["Roof - Main"])

    assert [node["id"] for node in filtered.values()] == [0, 1]
    assert {node["floor"] for node in filtered.values()} == {"First Floor", "Second Floor"}
