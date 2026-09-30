// Copies alphaTab's prebuilt browser files from node_modules into
// public/alphatab/ (git-ignored), run as `prebuild`. The sheet-music view
// loads the UMD script from there with a <script> tag instead of bundling it:
// alphaTab starts its render worker and audio worklet from its own script
// URL, which Next's webpack can't provide without alphaTab's webpack plugin.
// Also copies the Bravura music font (SIL OFL), the SONiVOX soundfont for the
// synth, and both licenses (alphaTab is MPL-2.0; shipped unmodified).
import { cpSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(root, "node_modules", "@coderline", "alphatab", "dist");
const out = join(root, "public", "alphatab");

rmSync(out, { recursive: true, force: true });
mkdirSync(join(out, "soundfont"), { recursive: true });
cpSync(join(dist, "alphaTab.min.js"), join(out, "alphaTab.min.js"));
cpSync(join(dist, "font"), join(out, "font"), { recursive: true });
cpSync(join(dist, "soundfont", "sonivox.sf2"), join(out, "soundfont", "sonivox.sf2"));
cpSync(join(dist, "soundfont", "LICENSE"), join(out, "soundfont", "LICENSE"));
cpSync(join(dist, "..", "LICENSE"), join(out, "LICENSE"));
console.log(`alphaTab assets copied to ${out}`);
