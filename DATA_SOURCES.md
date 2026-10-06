# IFC sources and evaluated graph provenance

Raw IFC inputs are obtained from their original sources rather than bundled in this repository. Prepared evaluated graphs and feature tables are included for reproducibility.

## Office

- Source model: KIT Institute, `AC20-Institute-Var-2.ifc`.
- [Source page at STEP Tools](https://www.steptools.com/docs/stpfiles/ifc/).
- Historical local filename: `Office Building.ifc`.
- Raw local SHA-256: `cf51ac647a45f4a73b15243f4acecd6d1833d33b21849d29d35f93b038d7b6f5`.
- Source comparison found a difference consisting only of the final newline.
- Evaluated graph: 82 spaces, 204 edges, five floors, preserved in `runs/Office_Building/data/processed/`.

The public source page is not treated as a grant to redistribute the raw Office model. Check its applicable terms with the source owner before redistributing raw inputs.

## Clinic

- Source attribution: BSI (2020), *Medical-Dental Test Files*, buildingSMART International.
- [Source and attribution](https://github.com/buildingsmart-community/Community-Sample-Test-Files/blob/main/IFC%202.3.0.1%20%28IFC%202x3%29/Medical-Dental%20Clinic/README.md).
- Source license: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
- Historical local filename: `Clinic_Architectural.ifc`.
- Raw SHA-256: `2ac970ce065ecac4e0c9e5f453a257169e90d0067f419b7e33533a64ef837880`; the source and local file matched byte for byte.
- Prepared graph changes: six spaces on `Roof - Main` excluded; IFC door opening width used in metres; node features and topology unchanged by the width correction.
- Evaluated graph: 263 spaces, 735 edges, two floors, preserved in `runs/Clinic_Architectural_width_corrected/data/processed/`.

The parser formerly read a 0.076 m frame width for 238 Clinic links. The paper and published result tree use corrected opening widths. The incorrect historical Clinic result tree is not included. The graph correction manifest and independent reference-distance audits accompany the results.

The IFC source license does not select a license for the authors' original project code. No new code license is selected by this publication preparation.
