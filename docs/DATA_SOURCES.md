# Building data

Prepared graphs and features are included. Obtain raw IFC files from the original sources below.

## Office

| Item | Value |
| --- | --- |
| Source model | KIT Institute, `AC20-Institute-Var-2.ifc` |
| Source | [STEP Tools sample files](https://www.steptools.com/docs/stpfiles/ifc/) |
| Original source and permission statement | [KIT IFC Examples](https://www.ifcwiki.org/index.php?title=KIT_IFC_Examples) |
| Publication attribution | Institute for Automation and Applied Informatics (IAI) / Karlsruhe Institute of Technology (KIT) |
| Local input name | `Office Building.ifc` |
| Evaluated graph | 82 spaces, 204 edges, five floors |
| Graph directory | `runs/Office_Building/data/processed/` |

Raw-file SHA-256: `cf51ac647a45f4a73b15243f4acecd6d1833d33b21849d29d35f93b038d7b6f5`.

Graph-file SHA-256: `786b90decb18cd7b466a68756b4c121db9eac76f99acd698ea308fc65fd706e4`.

Graph fingerprint used in model metadata: `bbb155bc3c33c01f61d9fecbe732a08ca2e80c61268ed169ca25a2f602e17339`.

The STEP Tools copy differs from the local file only by a final newline. The original KIT download also matches after normalizing CRLF/LF line endings and the terminal newline.

The [KIT examples page, revision 552](https://www.ifcwiki.org/index.php?title=KIT_IFC_Examples&oldid=552), checked on 2026-10-06, states that the examples are for unrestricted use and asks users to credit the Institute for Automation and Applied Informatics (IAI) / Karlsruhe Institute of Technology (KIT) in publications. This is the source's permission statement; the page does not name a Creative Commons license. Preserve this attribution and source link with any redistributed copy. This repository provides the derived graph and links to the raw IFC at its original source rather than including the raw file.

Original download: [AC20-Institute-Var-2.ifc](https://www.ifcwiki.org/images/9/98/AC20-Institute-Var-2.ifc). Its SHA-256 is `cfb2124497b25d9a72101075e84be0feb44ff669cb1bd3251be11efebeea945c`; the byte hash differs because of line endings.

## Clinic

| Item | Value |
| --- | --- |
| Attribution | BSI (2020), *Medical-Dental Test Files*, buildingSMART International |
| Source | [Model and attribution](https://github.com/buildingsmart-community/Community-Sample-Test-Files/blob/main/IFC%202.3.0.1%20%28IFC%202x3%29/Medical-Dental%20Clinic/README.md) |
| Source license | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Local input name | `Clinic_Architectural.ifc` |
| Evaluated graph | 263 spaces, 735 edges, two floors |
| Graph directory | `runs/Clinic_Architectural_width_corrected/data/processed/` |

Raw-file SHA-256: `2ac970ce065ecac4e0c9e5f453a257169e90d0067f419b7e33533a64ef837880`.

Graph-file SHA-256: `960adeb0893066dd99bdbd29d91aaa1499f1d9d83f64437e515e06b6f172de9f`.

Graph fingerprint used in model metadata: `e1b696035c275c3bee914d97ea22a4ff49ffec2937a7c1805c9eaa39eb50b9cc`.

The local raw file matched the source byte for byte. Six roof spaces were excluded. The evaluated graph uses IFC door opening widths rather than the former 0.076 m frame-width interpretation, which had incorrectly masked 238 links. The correction manifest and independent distance audit are in the Clinic result directory.
