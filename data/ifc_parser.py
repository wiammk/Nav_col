"""
data/ifc_parser.py
==================
Étape 1 du pipeline de données.

Lit un fichier IFC et extrait :
  - Les espaces (IfcSpace)         → nœuds  → nodes.csv
  - Les connexions (portes, escaliers, couloirs) → arêtes → edges.csv

Stratégie de connexion (3 niveaux) :
  1. IfcRelSpaceBoundary  : connexions via portes
  2. Escaliers            : connexions inter-étages adjacents UNIQUEMENT
  3. Proximité spatiale   : toujours appliquée en intra-étage

Usage :
    python data/ifc_parser.py --input data/raw_ifc/Office Building.ifc
    python data/ifc_parser.py --synthetic --floors 3 --rooms 8
"""

import os
import sys
import math
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.console import configure_console_encoding
from config.run_layout import RUNS_ROOT, RunLayout, sanitize_run_name

configure_console_encoding()

try:
    import ifcopenshell
    import ifcopenshell.geom
    from ifcopenshell.util.placement import get_local_placement
    _IFC_AVAILABLE = True
except ImportError:
    _IFC_AVAILABLE = False

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt= "%H:%M:%S"
)
log = logging.getLogger(__name__)

# ─── Chemins par défaut ───────────────────────────────────────────────────────
# ─── Constantes ───────────────────────────────────────────────────────────────
PROXIMITY_THRESHOLD = 8.0    # distance max (m) entre centroïdes pour edge de proximité
MAX_PROXIMITY_NEIGHBORS = 3  # garde les approximations géométriques locales
FLOOR_HEIGHT        = 4.0    # utilisée pour floor_index dans extract_spaces()
DOOR_FALLBACK_THRESHOLD = 12.0  # distance max (m) porte -> centroïde d'espace
RISK_CORRIDOR       = 0.10
RISK_ROOM           = 0.05
RISK_STORAGE        = 0.20
RISK_STAIR          = 0.15

SI_PREFIX_SCALE = {
    "EXA": 1e18,
    "PETA": 1e15,
    "TERA": 1e12,
    "GIGA": 1e9,
    "MEGA": 1e6,
    "KILO": 1e3,
    "HECTO": 1e2,
    "DECA": 1e1,
    "DECI": 1e-1,
    "CENTI": 1e-2,
    "MILLI": 1e-3,
    "MICRO": 1e-6,
    "NANO": 1e-9,
    "PICO": 1e-12,
    "FEMTO": 1e-15,
    "ATTO": 1e-18,
}


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 : UTILITAIRES IFC
# ─────────────────────────────────────────────────────────────────────────────

def get_length_unit_scale(ifc_file) -> float:
    """
    Retourne le multiplicateur qui convertit les coordonnées IFC en mètres.
    Exemple : MILLI METRE -> 0.001, METRE -> 1.0.
    """
    try:
        for assignment in ifc_file.by_type("IfcUnitAssignment"):
            for unit in assignment.Units:
                if getattr(unit, "UnitType", None) != "LENGTHUNIT":
                    continue
                if unit.is_a("IfcSIUnit"):
                    prefix = getattr(unit, "Prefix", None)
                    name = getattr(unit, "Name", None)
                    if name == "METRE":
                        return SI_PREFIX_SCALE.get(prefix, 1.0)
                if unit.is_a("IfcConversionBasedUnit"):
                    factor = getattr(unit, "ConversionFactor", None)
                    value = getattr(factor, "ValueComponent", None)
                    wrapped = getattr(value, "wrappedValue", None)
                    if wrapped:
                        return float(wrapped)
    except Exception as e:
        log.debug(f"Erreur lecture unités IFC : {e}")
    return 1.0


def get_local_placement_matrix(placement, unit_scale: float = 1.0) -> np.ndarray:
    """Return the complete world transform (translations + rotations) in metres."""
    matrix = np.eye(4, dtype=float)
    if placement is None:
        return matrix
    try:
        matrix = np.asarray(get_local_placement(placement), dtype=float).copy()
        matrix[:3, 3] *= unit_scale
    except Exception as e:
        log.debug(f"Erreur matrice de placement : {e}")
    return matrix


def get_local_placement_coords(placement, unit_scale: float = 1.0) -> Tuple[float, float, float]:
    matrix = get_local_placement_matrix(placement, unit_scale)
    return tuple(float(value) for value in matrix[:3, 3])


def get_element_world_coords(element, unit_scale: float = 1.0) -> Tuple[float, float, float]:
    """
    Return a representative world-space point in metres.

    IFC elements such as the KICT spaces may all share an ObjectPlacement whose
    X/Y translation is zero; their actual position is stored in their geometry.
    USE_WORLD_COORDS applies the complete placement matrix.  The bounding-box
    centre is stable for navigation and avoids tessellation-density bias.
    """
    if getattr(element, "Representation", None) is not None:
        try:
            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_WORLD_COORDS, True)
            shape = ifcopenshell.geom.create_shape(settings, element)
            vertices = np.asarray(shape.geometry.verts, dtype=float).reshape(-1, 3)
            if len(vertices):
                centre = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
                if np.isfinite(centre).all():
                    # IfcOpenShell geometry coordinates are already SI metres.
                    return tuple(float(value) for value in centre)
        except Exception as e:
            log.debug("Géométrie indisponible pour %s : %s", getattr(element, "GlobalId", "?"), e)
    return get_local_placement_coords(getattr(element, "ObjectPlacement", None), unit_scale)


def get_containing_storey(element):
    """Walk IFC spatial relations until the owning IfcBuildingStorey is found."""
    visited = set()
    queue = [element]
    while queue:
        current = queue.pop(0)
        current_id = current.id()
        if current_id in visited:
            continue
        visited.add(current_id)
        if current.is_a("IfcBuildingStorey"):
            return current
        relations = list(getattr(current, "Decomposes", ()) or ())
        relations += list(getattr(current, "ContainedInStructure", ()) or ())
        for relation in relations:
            parent = getattr(relation, "RelatingObject", None)
            if parent is None:
                parent = getattr(relation, "RelatingStructure", None)
            if parent is not None:
                queue.append(parent)
    return None


def get_storey_data(element, z: float, unit_scale: float = 1.0) -> Tuple[str, float, Optional[str]]:
    storey = get_containing_storey(element)
    if storey is not None:
        elevation = getattr(storey, "Elevation", None)
        if elevation is None:
            _, _, elevation_m = get_local_placement_coords(storey.ObjectPlacement, unit_scale)
        else:
            elevation_m = float(elevation) * unit_scale
        name = str(storey.Name or f"Storey_{storey.id()}")
        return name, elevation_m, getattr(storey, "GlobalId", None)
    floor_index = round(z / FLOOR_HEIGHT)
    return f"Floor_{floor_index}", floor_index * FLOOR_HEIGHT, None


def get_space_area(ifc_file, space) -> float:
    try:
        for rel in ifc_file.by_type("IfcRelDefinesByProperties"):
            try:
                if space not in rel.RelatedObjects:
                    continue
                prop_def = rel.RelatingPropertyDefinition
                if not prop_def.is_a("IfcElementQuantity"):
                    continue
                for q in prop_def.Quantities:
                    if q.is_a("IfcQuantityArea"):
                        name_lower = (q.Name or "").lower()
                        if any(k in name_lower for k in ["net", "floor", "plancher", "nette"]):
                            return float(q.AreaValue)
                for q in prop_def.Quantities:
                    if q.is_a("IfcQuantityArea"):
                        return float(q.AreaValue)
            except Exception:
                continue
    except Exception:
        pass
    return 25.0


def classify_space_type(space) -> str:
    name      = (getattr(space, "Name",     None) or "").lower()
    long_name = (getattr(space, "LongName", None) or "").lower()
    combined  = name + " " + long_name
    type_map  = {
        "corridor": ["corridor", "couloir", "hall", "hallway", "passage", "allée", "circulation"],
        "stair"   : ["stair", "escalier", "staircase", "stairwell", "stairway", "step"],
        "storage" : ["storage", "stockage", "warehouse", "entrepôt", "store", "depot", "dépôt", "atelier"],
        "office"  : ["office", "bureau", "work", "travail", "open space"],
        "toilet"  : ["toilet", "wc", "bathroom", "restroom", "sanitary", "sanitaire"],
        "hall"    : ["lobby", "reception", "accueil", "entrance", "entrée", "foyer"],
        "exit"    : ["exit", "sortie", "emergency", "secours"],
    }
    for space_type, keywords in type_map.items():
        if any(k in combined for k in keywords):
            return space_type
    return "room"


def get_initial_risk(space_type: str) -> float:
    return {
        "corridor": RISK_CORRIDOR,
        "stair"   : RISK_STAIR,
        "storage" : RISK_STORAGE,
        "room"    : RISK_ROOM,
        "office"  : RISK_ROOM,
        "toilet"  : RISK_ROOM,
        "hall"    : RISK_CORRIDOR,
        "exit"    : 0.0,
    }.get(space_type, RISK_ROOM)


def get_door_width(ifc_file, door, unit_scale: float = 1.0) -> float:
    """
    Read the opening width in metres, preferring IfcDoor.OverallWidth.

    A substring search also matches frame dimensions such as ``Trim Width``;
    those are not the width of the door opening. Property fallbacks therefore
    use exact semantic names. The default is already in metres.
    """
    def positive_metres(value):
        try:
            width = float(value) * float(unit_scale)
            return width if math.isfinite(width) and width > 0 else None
        except (TypeError, ValueError):
            return None

    overall = positive_metres(getattr(door, "OverallWidth", None))
    if overall is not None:
        return overall
    try:
        candidates = []
        names = {"overallwidth": 0, "doorwidth": 1, "clearwidth": 2, "width": 3}
        for rel in getattr(door, "IsDefinedBy", ()):
            if not rel.is_a("IfcRelDefinesByProperties"):
                continue
            prop_def = rel.RelatingPropertyDefinition
            if prop_def.is_a("IfcPropertySet"):
                for prop in prop_def.HasProperties:
                    name = "".join(c for c in (prop.Name or "").lower() if c.isalnum())
                    if name in names:
                        val = getattr(prop, "NominalValue", None)
                        width = positive_metres(getattr(val, "wrappedValue", None))
                        if width is not None:
                            candidates.append((names[name], width))
        if candidates:
            return min(candidates)[1]
    except Exception:
        pass
    return 0.9


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 : EXTRACTION DES NŒUDS
# ─────────────────────────────────────────────────────────────────────────────

def extract_spaces(ifc_file, unit_scale: float = 1.0) -> Dict[str, dict]:
    spaces      = ifc_file.by_type("IfcSpace")
    log.info(f"  {len(spaces)} espaces IfcSpace trouvés")
    spaces_data = {}
    skipped     = 0

    for idx, space in enumerate(spaces):
        try:
            guid = space.GlobalId
            name = str(space.Name or f"Space_{idx}")
            x, y, z = get_element_world_coords(space, unit_scale)
            area       = get_space_area(ifc_file, space)
            space_type = classify_space_type(space)
            floor_name, floor_elev, floor_guid = get_storey_data(space, z, unit_scale)

            spaces_data[guid] = {
                "id"        : idx,
                "guid"      : guid,
                "name"      : name,
                "x"         : round(x, 3),
                "y"         : round(y, 3),
                "z"         : round(z, 3),
                "area"      : round(area, 2),
                "type"      : space_type,
                "floor"     : floor_name,
                "floor_elev": round(floor_elev, 2),
                "floor_guid": floor_guid,
                "risk"      : get_initial_risk(space_type),
                "capacity"  : max(1, int(area / 10)),
            }
        except Exception as e:
            log.debug(f"Espace ignoré (idx={idx}) : {e}")
            skipped += 1

    log.info(f"  {len(spaces_data)} espaces extraits, {skipped} ignorés")
    return spaces_data


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 : EXTRACTION DES ARÊTES
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_distance(d1: dict, d2: dict) -> float:
    return math.sqrt(
        (d1["x"] - d2["x"])**2 +
        (d1["y"] - d2["y"])**2 +
        (d1["z"] - d2["z"])**2
    )


def distance_xy_to_point(space: dict, x: float, y: float) -> float:
    return math.sqrt((space["x"] - x)**2 + (space["y"] - y)**2)


def make_edge_key(edge: dict) -> Tuple[int, int]:
    return (min(edge["from"], edge["to"]), max(edge["from"], edge["to"]))


def infer_door_connections_by_geometry(
    ifc_file,
    doors: List,
    spaces_data: Dict,
    existing_edges: List[dict],
    unit_scale: float = 1.0,
    max_distance: float = DOOR_FALLBACK_THRESHOLD,
) -> List[dict]:
    """
    Fallback for IFC files that do not export usable IfcRelSpaceBoundary links.
    Associates each door with the two nearest spaces on the same estimated floor.
    """
    edges = []
    data_list = list(spaces_data.values())
    existing_keys = {make_edge_key(e) for e in existing_edges}

    for door in doors:
        try:
            dx, dy, dz = get_element_world_coords(door, unit_scale)
            door_name, door_floor, door_floor_guid = get_storey_data(door, dz, unit_scale)
            same_floor = [
                d for d in data_list
                if (
                    door_floor_guid is not None
                    and d.get("floor_guid") == door_floor_guid
                ) or (
                    door_floor_guid is None
                    and abs(d["floor_elev"] - door_floor) <= FLOOR_HEIGHT / 2
                )
            ]
            if len(same_floor) < 2:
                continue

            nearest = sorted(
                (
                    (distance_xy_to_point(space, dx, dy), space)
                    for space in same_floor
                ),
                key=lambda x: x[0],
            )[:2]

            if len(nearest) < 2 or nearest[1][0] > max_distance:
                continue

            d_from = nearest[0][1]
            d_to = nearest[1][1]
            edge = {
                "from"    : d_from["id"],
                "to"      : d_to["id"],
                "weight"  : round(euclidean_distance(d_from, d_to), 3),
                "type"    : "door",
                "passable": True,
                "width"   : get_door_width(ifc_file, door, unit_scale),
            }
            key = make_edge_key(edge)
            if key not in existing_keys:
                existing_keys.add(key)
                edges.append(edge)
        except Exception as e:
            log.debug(f"Fallback géométrique porte ignoré : {e}")

    return edges


def extract_door_connections(ifc_file, spaces_data: Dict, unit_scale: float = 1.0) -> List[dict]:
    """Connexions via IfcRelSpaceBoundary (portes)."""
    guid_to_id     = {g: d["id"] for g, d in spaces_data.items()}
    edges          = []
    found          = 0
    element_spaces : Dict[str, List[str]] = {}
    boundary_links = 0
    doors          = []

    try:
        doors = ifc_file.by_type("IfcDoor")
    except Exception as e:
        log.warning(f"Erreur lecture IfcDoor : {e}")

    door_guids = {door.GlobalId for door in doors if getattr(door, "GlobalId", None)}

    try:
        for rel in ifc_file.by_type("IfcRelSpaceBoundary"):
            try:
                if rel.RelatedBuildingElement is None or rel.RelatingSpace is None:
                    continue
                elem_guid  = rel.RelatedBuildingElement.GlobalId
                space_guid = rel.RelatingSpace.GlobalId
                if elem_guid not in door_guids:
                    continue
                if space_guid not in guid_to_id:
                    continue
                if elem_guid not in element_spaces:
                    element_spaces[elem_guid] = []
                if space_guid not in element_spaces[elem_guid]:
                    element_spaces[elem_guid].append(space_guid)
                    boundary_links += 1
            except Exception:
                continue
    except Exception as e:
        log.warning(f"Erreur IfcRelSpaceBoundary : {e}")

    try:
        for door in doors:
            guids = element_spaces.get(door.GlobalId, [])
            valid = [g for g in guids if g in guid_to_id]
            if len(valid) >= 2:
                for i in range(len(valid)):
                    for j in range(i + 1, len(valid)):
                        d_from = spaces_data[valid[i]]
                        d_to   = spaces_data[valid[j]]
                        edges.append({
                            "from"    : d_from["id"],
                            "to"      : d_to["id"],
                            "weight"  : round(euclidean_distance(d_from, d_to), 3),
                            "type"    : "door",
                            "passable": True,
                            "width"   : get_door_width(ifc_file, door, unit_scale),
                        })
                        found += 1
    except Exception as e:
        log.warning(f"Erreur extraction portes : {e}")

    log.info(
        "  Portes IFC : %s portes, %s relations espace-porte, %s connexions directes",
        len(doors),
        boundary_links,
        found,
    )

    # IFC exporters are often inconsistent per door: a good global boundary
    # count can still leave individual rooms disconnected.  Apply geometry only
    # to doors that do not already expose two usable space boundaries.
    fallback_doors = [
        door for door in doors
        if len(element_spaces.get(getattr(door, "GlobalId", None), [])) < 2
    ]
    if fallback_doors:
        fallback_edges = infer_door_connections_by_geometry(
            ifc_file,
            fallback_doors,
            spaces_data,
            edges,
            unit_scale,
        )
        edges.extend(fallback_edges)
        log.info(
            "  Fallback géométrique portes : %s connexions ajoutées "
            "pour %s portes incomplètes (seuil %.1fm)",
            len(fallback_edges),
            len(fallback_doors),
            DOOR_FALLBACK_THRESHOLD,
        )

    log.info(f"  {len(edges)} connexions par portes conservées")
    return edges


def extract_stair_connections(ifc_file, spaces_data: Dict, unit_scale: float = 1.0) -> List[dict]:
    """
    Connexions inter-étages entre IfcBuildingStorey adjacents.

    A stair flight belongs to its departure storey.  Its world-space geometry
    locates the stair shaft; the two closest spaces on the departure and next
    represented storeys become the navigation connection.  This avoids the old
    fixed 4 m floor-height assumption (KICT has a 5.2 m basement transition).
    """
    try:
        flights = list(ifc_file.by_type("IfcStairFlight"))
        stair_elements = flights or list(ifc_file.by_type("IfcStair"))
        stair_elements += list(ifc_file.by_type("IfcRamp"))
    except Exception:
        stair_elements = []

    # by_type may include subtypes; never process an IFC entity twice.
    stair_elements = list({element.id(): element for element in stair_elements}.values())
    if not stair_elements:
        log.info("  Aucun escalier trouvé dans le fichier IFC")
        return []

    by_floor = defaultdict(list)
    for space in spaces_data.values():
        floor_key = space.get("floor_guid") or space["floor"]
        by_floor[floor_key].append(space)
    ordered_floors = sorted(
        by_floor,
        key=lambda key: min(space["floor_elev"] for space in by_floor[key]),
    )
    floor_index = {key: idx for idx, key in enumerate(ordered_floors)}
    edges = []
    skipped_no_floor = 0
    skipped_top_floor = 0

    for stair in stair_elements:
        try:
            sx, sy, sz = get_element_world_coords(stair, unit_scale)
            _, _, storey_guid = get_storey_data(stair, sz, unit_scale)
            floor_key = storey_guid
            if floor_key not in floor_index:
                # Fallback only for malformed IFCs without a spatial relation.
                floor_key = min(
                    ordered_floors,
                    key=lambda key: abs(by_floor[key][0]["floor_elev"] - sz),
                )
            idx = floor_index[floor_key]
            if idx + 1 >= len(ordered_floors):
                skipped_top_floor += 1
                continue

            next_floor_key = ordered_floors[idx + 1]
            lower = min(by_floor[floor_key], key=lambda d: distance_xy_to_point(d, sx, sy))
            upper = min(by_floor[next_floor_key], key=lambda d: distance_xy_to_point(d, sx, sy))
            edges.append({
                "from": lower["id"],
                "to": upper["id"],
                "weight": round(euclidean_distance(lower, upper), 3),
                "type": "stair",
                "passable": True,
                "width": 1.2,
            })
        except Exception as e:
            log.debug(f"Erreur escalier : {e}")
            skipped_no_floor += 1

    unique_edges = deduplicate_edges(edges)
    log.info(
        "  %s connexions inter-étages candidates, %s uniques "
        "(%s éléments au dernier étage, %s ignorés)",
        len(edges),
        len(unique_edges),
        skipped_top_floor,
        skipped_no_floor,
    )
    return unique_edges


def extract_proximity_connections(
    spaces_data    : Dict,
    existing_edges : List[dict],
    threshold      : float = PROXIMITY_THRESHOLD,
    max_neighbors  : int = MAX_PROXIMITY_NEIGHBORS,
) -> List[dict]:
    """
    Connexions intra-étage entre espaces proches non encore connectés.
    Garantit un degré moyen suffisant pour le RL.
    """
    edges     = []
    connected = set()

    for e in existing_edges:
        connected.add((min(e["from"], e["to"]), max(e["from"], e["to"])))

    data_list = list(spaces_data.values())

    # Keep only the closest geometric fallbacks for every space. This avoids
    # turning rooms that merely share a floor into artificial corridors.
    candidates = []
    choices = {space["id"]: [] for space in data_list}
    for i in range(len(data_list)):
        for j in range(i + 1, len(data_list)):
            d1, d2 = data_list[i], data_list[j]
            pair = (min(d1["id"], d2["id"]), max(d1["id"], d2["id"]))
            if pair in connected or d1["floor"] != d2["floor"]:
                continue
            distance = euclidean_distance(d1, d2)
            if distance > threshold:
                continue
            edge = {
                "from": d1["id"],
                "to": d2["id"],
                "weight": round(distance, 3),
                "type": "proximity",
                "passable": True,
                "width": 1.5,
            }
            candidates.append(edge)
            choices[d1["id"]].append(edge)
            choices[d2["id"]].append(edge)

    selected = set()
    for node_id, node_choices in choices.items():
        nearest = sorted(
            node_choices,
            key=lambda edge: (edge["weight"], edge["to"] if edge["from"] == node_id else edge["from"]),
        )
        for edge in nearest[:max_neighbors]:
            selected.add(make_edge_key(edge))

    edges = [edge for edge in candidates if make_edge_key(edge) in selected]
    log.info(
        "  %s connexions de proximite ajoutees (seuil=%sm, max_voisins=%s)",
        len(edges),
        threshold,
        max_neighbors,
    )
    return edges

    for i in range(len(data_list)):
        for j in range(i + 1, len(data_list)):
            d1, d2 = data_list[i], data_list[j]
            pair   = (min(d1["id"], d2["id"]), max(d1["id"], d2["id"]))

            if pair in connected:
                continue
            if d1["floor"] != d2["floor"]:
                continue

            dist = euclidean_distance(d1, d2)
            if dist <= threshold:
                edges.append({
                    "from"    : d1["id"],
                    "to"      : d2["id"],
                    "weight"  : round(dist, 3),
                    "type"    : "proximity",
                    "passable": True,
                    "width"   : 1.5,
                })
                connected.add(pair)

    log.info(f"  {len(edges)} connexions de proximité ajoutées")
    return edges


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 : SAUVEGARDE CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_nodes_csv(spaces_data: Dict, path: Path) -> pd.DataFrame:
    rows = [
        {
            "id"        : d["id"],
            "guid"      : d["guid"],
            "name"      : d["name"],
            "x"         : d["x"],
            "y"         : d["y"],
            "z"         : d["z"],
            "area"      : d["area"],
            "type"      : d["type"],
            "floor"     : d["floor"],
            "floor_elev": d["floor_elev"],
            "floor_guid": d.get("floor_guid"),
            "risk"      : d["risk"],
            "capacity"  : d["capacity"],
        }
        for d in spaces_data.values()
    ]
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    df.to_csv(path, index=False)
    log.info(f"  ✅ nodes.csv sauvegardé : {len(df)} nœuds → {path}")
    return df


def deduplicate_edges(edges: List[dict]) -> List[dict]:
    seen, unique = set(), []
    for e in edges:
        key = (min(e["from"], e["to"]), max(e["from"], e["to"]))
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def select_single_floor(spaces_data: Dict, selection: str) -> Dict:
    """Keep one real IFC storey and reindex its nodes for a safe 2D experiment."""
    floors = {}
    for guid, node in spaces_data.items():
        key = node.get("floor_guid") or node.get("floor")
        floors.setdefault(key, []).append((guid, node))

    if selection.lower() == "largest":
        selected_key, selected_nodes = max(floors.items(), key=lambda item: len(item[1]))
    else:
        matches = [
            (key, nodes) for key, nodes in floors.items()
            if str(key).casefold() == selection.casefold()
            or str(nodes[0][1].get("floor", "")).casefold() == selection.casefold()
        ]
        if not matches:
            available = sorted({str(nodes[0][1].get("floor", key)) for key, nodes in floors.items()})
            raise ValueError(
                f"Etage IFC introuvable: {selection!r}. Etages disponibles: {', '.join(available)}"
            )
        selected_key, selected_nodes = matches[0]

    filtered = {}
    for new_id, (guid, original) in enumerate(selected_nodes):
        node = dict(original)
        node["id"] = new_id
        filtered[guid] = node
    floor_name = selected_nodes[0][1].get("floor", selected_key)
    log.info(
        "  Mode mono-etage : %s (%s espaces sur %s) conserve",
        floor_name, len(filtered), len(spaces_data),
    )
    return filtered


def exclude_floors(spaces_data: Dict, selections: List[str]) -> Dict:
    """Exclude explicitly named non-navigable IFC storeys and reindex nodes."""
    requested = {str(value).casefold() for value in selections if str(value).strip()}
    if not requested:
        return spaces_data

    available = {
        str(node.get("floor", "")).casefold(): str(node.get("floor", ""))
        for node in spaces_data.values()
    }
    available.update({
        str(node.get("floor_guid", "")).casefold(): str(node.get("floor", ""))
        for node in spaces_data.values() if node.get("floor_guid")
    })
    missing = sorted(selection for selection in requested if selection not in available)
    if missing:
        floor_names = sorted({str(node.get("floor", "")) for node in spaces_data.values()})
        raise ValueError(
            "Etage(s) IFC a exclure introuvable(s): "
            + ", ".join(missing)
            + ". Etages disponibles: "
            + ", ".join(floor_names)
        )

    kept_items = [
        (guid, node) for guid, node in spaces_data.items()
        if str(node.get("floor", "")).casefold() not in requested
        and str(node.get("floor_guid", "")).casefold() not in requested
    ]
    if not kept_items:
        raise ValueError("L'exclusion des etages supprimerait tous les espaces IFC")

    filtered = {}
    for new_id, (guid, original) in enumerate(
        sorted(kept_items, key=lambda item: item[1]["id"])
    ):
        node = dict(original)
        node["id"] = new_id
        filtered[guid] = node

    excluded_names = sorted({available[selection] for selection in requested})
    log.info(
        "  Etage(s) non navigable(s) exclu(s) : %s (%s espaces conserves sur %s)",
        ", ".join(excluded_names), len(filtered), len(spaces_data),
    )
    return filtered


def validate_spatial_data(
    spaces_data: Dict,
    edges: List[dict],
    max_origin_ratio: float = 0.10,
    max_zero_distance_ratio: float = 0.05,
    min_unique_xy_ratio: float = 0.25,
) -> dict:
    """Reject degenerate IFC graphs before they can produce misleading RL results."""
    nodes = list(spaces_data.values())
    if not nodes:
        raise ValueError("Graphe IFC invalide : aucun nœud")

    coords = np.asarray([[n["x"], n["y"], n["z"]] for n in nodes], dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("Graphe IFC invalide : coordonnées non finies")

    origin_count = int(np.sum(np.isclose(coords[:, 0], 0.0) & np.isclose(coords[:, 1], 0.0)))
    unique_xy = len({(round(float(x), 6), round(float(y), 6)) for x, y in coords[:, :2]})
    node_by_id = {node["id"]: node for node in nodes}
    zero_edges = sum(
        1 for edge in edges
        if edge["from"] in node_by_id
        and edge["to"] in node_by_id
        and euclidean_distance(node_by_id[edge["from"]], node_by_id[edge["to"]]) <= 1e-6
    )

    origin_ratio = origin_count / len(nodes)
    unique_xy_ratio = unique_xy / len(nodes)
    zero_distance_ratio = zero_edges / len(edges) if edges else 0.0
    errors = []
    if origin_ratio > max_origin_ratio:
        errors.append(f"{origin_ratio:.1%} des nœuds ont x=y=0 (maximum {max_origin_ratio:.1%})")
    if unique_xy_ratio < min_unique_xy_ratio:
        errors.append(
            f"seulement {unique_xy}/{len(nodes)} positions XY distinctes "
            f"({unique_xy_ratio:.1%}, minimum {min_unique_xy_ratio:.1%})"
        )
    if zero_distance_ratio > max_zero_distance_ratio:
        errors.append(
            f"{zero_edges}/{len(edges)} arêtes ont une distance nulle "
            f"({zero_distance_ratio:.1%}, maximum {max_zero_distance_ratio:.1%})"
        )

    floor_key_by_id = {
        node["id"]: node.get("floor_guid") or node["floor"]
        for node in nodes
    }
    floor_elevations = {
        node.get("floor_guid") or node["floor"]: node["floor_elev"]
        for node in nodes
    }
    invalid_horizontal = [
        edge for edge in edges
        if edge.get("type", "proximity") in {"door", "proximity", "corridor"}
        and floor_key_by_id[edge["from"]] != floor_key_by_id[edge["to"]]
    ]
    if invalid_horizontal:
        errors.append(f"{len(invalid_horizontal)} connexions horizontales traversent des étages")

    ordered_floors = sorted(floor_elevations, key=floor_elevations.get)
    stair_pairs = {
        frozenset((floor_key_by_id[edge["from"]], floor_key_by_id[edge["to"]]))
        for edge in edges if edge.get("type") == "stair"
    }
    missing_floor_links = [
        (ordered_floors[idx], ordered_floors[idx + 1])
        for idx in range(len(ordered_floors) - 1)
        if frozenset((ordered_floors[idx], ordered_floors[idx + 1])) not in stair_pairs
    ]
    if missing_floor_links:
        errors.append(f"liaisons d'escaliers absentes entre {len(missing_floor_links)} étages adjacents")
    if errors:
        raise ValueError("Graphe IFC dégénéré : " + " ; ".join(errors))

    report = {
        "nodes": len(nodes),
        "edges": len(edges),
        "origin_xy_ratio": origin_ratio,
        "unique_xy_ratio": unique_xy_ratio,
        "zero_distance_ratio": zero_distance_ratio,
        "floor_count": len(ordered_floors),
        "adjacent_floor_links": max(0, len(ordered_floors) - 1),
    }
    log.info("  Validation spatiale réussie : %s", report)
    return report


def save_edges_csv(edges: List[dict], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(edges, columns=["from", "to", "weight", "type", "passable", "width"])
    df = df.sort_values(["from", "to"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    log.info(f"  ✅ edges.csv sauvegardé : {len(df)} arêtes → {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 : GÉNÉRATEUR SYNTHÉTIQUE
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticBuildingGenerator:
    def __init__(self, n_floors=2, rooms_per_floor=6,
                 floor_height=FLOOR_HEIGHT, room_spacing=12.0, seed=42):
        self.n_floors        = n_floors
        self.rooms_per_floor = rooms_per_floor
        self.floor_height    = floor_height
        self.room_spacing    = room_spacing
        np.random.seed(seed)
        self.room_types = ["room", "room", "room", "office", "storage", "corridor"]

    def generate(self) -> Tuple[Dict, List[dict]]:
        spaces_data = {}
        edges       = []
        node_id     = 0
        cols        = max(3, int(math.ceil(math.sqrt(self.rooms_per_floor))))
        floor_nodes = {}

        for floor_idx in range(self.n_floors):
            floor_name = f"Floor_{floor_idx}"
            floor_elev = floor_idx * self.floor_height
            floor_nodes[floor_idx] = []

            for room_idx in range(self.rooms_per_floor):
                row = room_idx // cols
                col = room_idx  % cols
                x   = col * self.room_spacing + np.random.uniform(-1, 1)
                y   = row * self.room_spacing + np.random.uniform(-1, 1)
                z   = floor_elev

                if room_idx == 0 and floor_idx == 0:
                    space_type = "hall"
                elif room_idx == self.rooms_per_floor - 1:
                    space_type = "exit"
                elif room_idx % cols == 0:
                    space_type = "corridor"
                else:
                    space_type = self.room_types[room_idx % len(self.room_types)]

                area = (np.random.uniform(20, 150) if space_type != "corridor"
                        else np.random.uniform(8, 30))

                guid = f"SYNTH_{floor_idx}_{room_idx:04d}"
                data = {
                    "id"        : node_id,
                    "guid"      : guid,
                    "name"      : f"{space_type.title()}_{floor_idx}_{room_idx}",
                    "x"         : round(x, 3),
                    "y"         : round(y, 3),
                    "z"         : round(z, 3),
                    "area"      : round(area, 2),
                    "type"      : space_type,
                    "floor"     : floor_name,
                    "floor_elev": round(floor_elev, 2),
                    "risk"      : get_initial_risk(space_type),
                    "capacity"  : max(1, int(area / 10)),
                }
                spaces_data[guid] = data
                floor_nodes[floor_idx].append(data)
                node_id += 1

        for floor_idx, nodes in floor_nodes.items():
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    d1, d2 = nodes[i], nodes[j]
                    dist   = euclidean_distance(d1, d2)
                    if dist <= self.room_spacing * 1.5:
                        conn_type = (
                            "stair"    if d1["type"] == "stair" or d2["type"] == "stair"
                            else "corridor" if d1["type"] == "corridor" or d2["type"] == "corridor"
                            else "door"
                        )
                        edges.append({
                            "from"    : d1["id"],
                            "to"      : d2["id"],
                            "weight"  : round(dist, 3),
                            "type"    : conn_type,
                            "passable": True,
                            "width"   : 1.2 if conn_type == "door" else 2.0,
                        })

        for floor_idx in range(self.n_floors - 1):
            nodes_low  = floor_nodes[floor_idx]
            nodes_high = floor_nodes[floor_idx + 1]
            best_pairs = sorted(
                ((math.sqrt((d1["x"]-d2["x"])**2 + (d1["y"]-d2["y"])**2), d1, d2)
                 for d1 in nodes_low for d2 in nodes_high),
                key=lambda x: x[0]
            )
            for _, d1, d2 in best_pairs[:2]:
                edges.append({
                    "from"    : d1["id"],
                    "to"      : d2["id"],
                    "weight"  : round(euclidean_distance(d1, d2), 3),
                    "type"    : "stair",
                    "passable": True,
                    "width"   : 1.5,
                })

        log.info(f"  ---Bâtiment synthétique--- : {len(spaces_data)} nœuds, "
                 f"{len(edges)} arêtes, {self.n_floors} étages")
        return spaces_data, edges


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 : PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def parse_ifc(
    ifc_path: str,
    proximity_threshold: float = PROXIMITY_THRESHOLD,
    max_proximity_neighbors: int = MAX_PROXIMITY_NEIGHBORS,
    single_floor: str = None,
    excluded_floors: List[str] = None,
) -> Tuple[Dict, List]:
    if not _IFC_AVAILABLE:
        log.error("ifcopenshell non installé. pip install ifcopenshell")
        sys.exit(1)

    log.info(f" ---Chargement IFC ---: {ifc_path}")
    ifc_file = ifcopenshell.open(ifc_path)
    log.info(f"  Schéma IFC : {ifc_file.schema}")
    unit_scale = get_length_unit_scale(ifc_file)
    log.info(f"  Unité longueur IFC : 1 unité = {unit_scale:g} m")

    log.info(" ---Extraction des espaces---")
    spaces_data = extract_spaces(ifc_file, unit_scale)
    if not spaces_data:
        log.warning("Aucun espace trouvé !")
        return {}, []

    if single_floor and excluded_floors:
        raise ValueError("Utilisez soit single_floor, soit excluded_floors, pas les deux")
    if single_floor:
        spaces_data = select_single_floor(spaces_data, single_floor)
    elif excluded_floors:
        spaces_data = exclude_floors(spaces_data, excluded_floors)

    log.info(" ---Extraction des connexions par portes---")
    door_edges = extract_door_connections(ifc_file, spaces_data, unit_scale)

    log.info(" ---Extraction des connexions par escaliers---")
    stair_edges = extract_stair_connections(ifc_file, spaces_data, unit_scale)

    all_edges = door_edges + stair_edges
    stair_candidates = len(stair_edges)

    coverage = len(
        set(e["from"] for e in all_edges) |
        set(e["to"]   for e in all_edges)
    )
    log.info(f"  Nœuds connectés après portes+escaliers : {coverage}/{len(spaces_data)}")

    log.info(" ---Ajout des connexions de proximité intra-étage---")
    # Note : threshold ici utilise la valeur par défaut PROXIMITY_THRESHOLD
    # Pour passer une valeur custom, utiliser extract_proximity_connections() directement
    prox_edges = extract_proximity_connections(
        spaces_data,
        all_edges,
        threshold=proximity_threshold,
        max_neighbors=max_proximity_neighbors,
    )
    all_edges += prox_edges

    raw_type_counts = Counter(e["type"] for e in all_edges)
    raw_edge_count = len(all_edges)
    all_edges = deduplicate_edges(all_edges)
    final_type_counts = Counter(e["type"] for e in all_edges)
    removed_count = raw_edge_count - len(all_edges)
    if removed_count:
        log.info(
            "  Déduplication : %s arêtes candidates fusionnées "
            "(avant=%s, final=%s)",
            removed_count,
            dict(raw_type_counts),
            dict(final_type_counts),
        )
    if stair_candidates != final_type_counts.get("stair", 0):
        log.info(
            "  Escaliers : %s candidates détectées, %s arêtes stair uniques conservées",
            stair_candidates,
            final_type_counts.get("stair", 0),
        )
    log.info(f"  Total arêtes finales : {len(all_edges)}")

    validate_spatial_data(spaces_data, all_edges)

    return spaces_data, all_edges


def run_synthetic(n_floors: int = 2, rooms: int = 6) -> Tuple[Dict, List]:
    log.info(f" ---Génération synthétique--- : {n_floors} étages, {rooms} salles/étage")
    return SyntheticBuildingGenerator(n_floors=n_floors, rooms_per_floor=rooms).generate()


def main():
    parser = argparse.ArgumentParser(description="IFC Parser")
    parser.add_argument("--input",      "-i", type=str, default=None)
    parser.add_argument("--synthetic",  "-s", action="store_true")
    parser.add_argument("--floors",     type=int,   default=2)
    parser.add_argument("--rooms",      type=int,   default=6)
    parser.add_argument("--proximity",  type=float, default=PROXIMITY_THRESHOLD)
    parser.add_argument("--max-proximity-neighbors", type=int, default=MAX_PROXIMITY_NEIGHBORS)
    parser.add_argument("--single-floor", default=None, metavar="ETAGE",
                        help="Limiter l'experience a un IfcBuildingStorey (nom/GUID, ou 'largest').")
    parser.add_argument(
        "--exclude-floor", action="append", default=None, metavar="ETAGE",
        help="Exclure explicitement un etage IFC non navigable (option repetable).",
    )
    parser.add_argument("--verbose",    "-v", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.input:
        out_dir = RunLayout(RUNS_ROOT / sanitize_run_name(Path(args.input).stem)).processed
    else:
        name = f"synthetic_{args.floors}floors_{args.rooms}rooms"
        out_dir = RunLayout(RUNS_ROOT / name).processed
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        spaces_data, edges = run_synthetic(args.floors, args.rooms)
    elif args.input:
        spaces_data, edges = parse_ifc(
            args.input,
            proximity_threshold=args.proximity,
            max_proximity_neighbors=args.max_proximity_neighbors,
            single_floor=args.single_floor,
            excluded_floors=args.exclude_floor,
        )
    else:
        log.error("Spécifiez --input <fichier.ifc> ou --synthetic")
        parser.print_help()
        sys.exit(1)

    if not spaces_data:
        log.error("Aucune donnée extraite. Arrêt.")
        sys.exit(1)

    if args.synthetic:
        pass  # déjà calculé dans SyntheticBuildingGenerator
    elif args.input:
        # Pour parse_ifc, re-calculer la proximité avec le seuil custom si différent
        if False:
            log.info(f"--- Recalcul proximité avec seuil custom --- : {args.proximity}m")
            door_stair = [e for e in edges if e["type"] != "corridor"]
            prox       = extract_proximity_connections(spaces_data, door_stair, args.proximity)
            edges      = deduplicate_edges(door_stair + prox)

    df_nodes = save_nodes_csv(spaces_data, out_dir / "nodes.csv")
    df_edges = save_edges_csv(edges,       out_dir / "edges.csv")

    log.info("=" * 55)
    log.info(" ---RÉSUMÉ DU PARSING---")
    log.info("=" * 55)
    log.info(f"  Nœuds          : {len(df_nodes)}")
    log.info(f"  Arêtes         : {len(df_edges)}")
    log.info(f"  Types nœuds    : {df_nodes['type'].value_counts().to_dict()}")
    log.info(f"  Types arêtes   : {df_edges['type'].value_counts().to_dict()}")
    log.info(f"  Étages         : {df_nodes['floor'].unique().tolist()}")
    avg_deg = len(df_edges) * 2 / len(df_nodes)
    log.info(f"  Degré moyen    : {avg_deg:.2f}")
    log.info("=" * 55)
    log.info("✅ Parsing terminé. Lancez : python data/graph_builder.py")


if __name__ == "__main__":
    main()
