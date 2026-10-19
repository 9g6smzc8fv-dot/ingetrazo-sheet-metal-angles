# Sheet Metal Angles for IngeTrazo

Editable sheet-metal pans, panels, flanges and returns with adjustable bend angles, flat patterns and DXF export.

**Version 3.5.0 · Tested with IngeTrazo 0.5.8 · MIT license**

Created by Kevin Berg with AI assistance. Feedback and example results are welcome in [Issues](https://github.com/9g6smzc8fv-dot/ingetrazo-sheet-metal-angles/issues) or the [IngeTrazo discussion](https://github.com/ingelibre/ingetrazo/discussions/684).

## Features

- Independent flange and return bend angles from 1 to 170 degrees, with 90 degrees as the default.
- Inches and millimetres selectable in the tool.
- Stainless and cold-rolled steel presets, with editable thickness, inside radius and bend deduction.
- Formed and flat views, bend lines, angle notes and DXF export.
- Editable openings, holes, slots and cuts, with Undo/Redo support.
- Loads definitions from earlier REM unified, guided and cab-panel tools.

![Example part with a 75-degree flange and a 45-degree return](https://github.com/user-attachments/assets/55eb88ad-167a-44c9-9e39-d6f6f0c29dcd)

![Bend-angle control](https://github.com/user-attachments/assets/3666e28e-f543-4dd6-bc41-3a8604e6157f)

## Install

1. Download **Sheet-Metal-Angles-3.5.0.zip** from [Releases](https://github.com/9g6smzc8fv-dot/ingetrazo-sheet-metal-angles/releases).
2. Extract the ZIP to a normal folder.
3. In IngeTrazo, choose **Extensions → Open plugins folder** (the wording may be Open extensions folder).
4. Save your drawings and close IngeTrazo.
5. Copy **only `sheet_metal_angles.py`** into the plugins folder.
6. Reopen IngeTrazo and choose **Extensions → Sheet Metal Angles**, or press **Ctrl+Alt+B** if that shortcut is available.

If upgrading an earlier 3.5 installation, remove the old **rem_sheet_metal_angles.py** file while IngeTrazo is closed and keep only **sheet_metal_angles.py**. Existing part definitions remain compatible.

No separate Python installation is needed. The original Sheet Metal tool can remain installed separately.

Once the community catalog submission is accepted, the extension will also be available through **Extensions → Manage extensions**.

## Use

Start with **New Base**, set the size and units, place the base, then choose **Flange** and select an edge. Returns have their own angle setting. To edit an existing generated part, select the whole part, choose **Load Selected**, and double-click its feature.

Angles are **rotation from flat**:

| Bend setting | Opening angle |
| --- | --- |
| 45° | 135° |
| 90° | 90° |
| 135° | 45° |

Flange leg length is measured along the outside leg to the virtual sharp, rather than as the vertical height of an angled flange.

## Current limits

- **Non-90-degree bends require Open corners.** Closed, full-width, mitred and tabbed corner constructions use the existing 90-degree algorithms.
- Units are chosen in the tool; automatic adoption of the document's display units is not implemented.
- Flat development at other angles uses a neutral radius calibrated from the material's 90-degree bend data. Check a sample bend with your shop before production cutting. Springback and press-brake tool compensation are not applied.
- Older tools cannot edit parts saved under this extension's new angled definition. Use Sheet Metal Angles 3.5 on computers editing those parts.

See [USAGE.txt](USAGE.txt) for the detailed development equations, compatibility notes and workflow.

## Examples and validation

The release ZIP includes generic formed and flat `.igz` files and a DXF for a 12 × 18 inch base, a 75-degree UP flange and a 45-degree inward return.

Validation performed for this version: 720 numerical angle combinations, 80 native closed-solid builds, legacy 90-degree geometry comparisons, and isolated native UI checks for editing, Undo/Redo, Cancel, flat/formed switching and save/reopen. These are software checks; no physical forming trial was run.

## License

[MIT](LICENSE). Copyright © 2026 Kevin Berg.
