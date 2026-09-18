# Vendored fonts

These power the hand-drawn `sketch` slide style. They are committed to the repo
(rather than fetched from Google Fonts at render time) so that:

* the sketch style renders identically with no network access, and
* a saved `slide_NN.html` stays portable — the bytes are embedded in the file.

Only the **latin subset** of each face is included, which is why they are small.

| File | Family | Weight | Size |
|---|---|---|---|
| `Caveat-700.woff2` | Caveat | 700 | ~50 KB |
| `Kalam-400.woff2` | Kalam | 400 | ~22 KB |
| `Kalam-700.woff2` | Kalam | 700 | ~22 KB |

## Licence

Both families are licensed under the **SIL Open Font License 1.1**, which permits
bundling and redistribution with software.

* Caveat — © Impallari Type. <https://fonts.google.com/specimen/Caveat>
* Kalam — © Indian Type Foundry. <https://fonts.google.com/specimen/Kalam>

Full licence text: <https://scripts.sil.org/OFL>

## Re-vendoring

If you want different handwriting fonts, drop the woff2 files here and update
`_SKETCH_FACES` plus the `hand_display_family` / `hand_body_family` stacks in
`app/rendering/design_system.py`. `fonts_available()` reports whether the files
the code expects are actually present; when they are missing the stacks fall
back to a system cursive face rather than failing the render.
