"""Record actual IFC width attributes and the property previously selected."""
import json
import pickle
from collections import Counter
from pathlib import Path
import ifcopenshell
import ifcopenshell.util.unit
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
records = {}
for name, file in [('clinic', 'Clinic_Architectural.ifc'), ('office', 'Office Building.ifc')]:
    model = ifcopenshell.open(str(ROOT / 'data/raw_ifc' / file))
    scale = ifcopenshell.util.unit.calculate_unit_scale(model)
    linked_properties = {}
    for rel in model.by_type('IfcRelDefinesByProperties'):
        for obj in rel.RelatedObjects:
            if obj.is_a('IfcDoor'):
                linked_properties.setdefault(obj.id(), []).append(rel.RelatingPropertyDefinition)
    doors = []
    for door in model.by_type('IfcDoor'):
        properties = []
        for pset in linked_properties.get(door.id(), []):
            if pset.is_a('IfcPropertySet'):
                for prop in pset.HasProperties:
                    if 'width' in (prop.Name or '').lower():
                        val = getattr(prop, 'NominalValue', None)
                        properties.append({'set': pset.Name, 'name': prop.Name,
                            'value': getattr(val, 'wrappedValue', None)})
        doors.append({'id': door.id(), 'guid': door.GlobalId,
            'overall_width': door.OverallWidth, 'width_properties': properties})
    directory = 'Clinic_Architectural' if name == 'clinic' else 'Office_Building'
    with (ROOT / 'runs' / directory / 'data/processed/graph.gpickle').open('rb') as f:
        graph = pickle.load(f)
    legal = graph.copy()
    illegal = [(u, v) for u, v, d in graph.edges(data=True)
               if not d.get('passable', True) or d.get('width', 1.) < .6]
    legal.remove_edges_from(illegal)
    records[name] = {'unit_scale': scale, 'doors': doors,
        'legacy_first_width_properties': dict(Counter(
            d['width_properties'][0]['name'] for d in doors if d['width_properties'])),
        'narrow_edges': sum(d.get('width', 1.) < .6 for u, v, d in graph.edges(data=True)),
        'legal_component_sizes': sorted(map(len, nx.connected_components(legal)), reverse=True),
        'legacy_width_values': [{'type': k[0], 'width': k[1], 'count': v}
            for k, v in Counter((d.get('type'), d.get('width')) for u,v,d in graph.edges(data=True)).items()]}
out = ROOT / 'runs/Office_Building/review_followup/ifc_door_width_audit.json'
out.write_text(json.dumps(records, indent=2), encoding='utf-8')
print(json.dumps({k: {n: v for n,v in data.items() if n != 'doors'}
                  for k,data in records.items()}, ensure_ascii=True))
