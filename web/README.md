# NightRun website (`web/`)

The project's static site: four plain HTML pages, one stylesheet, one tiny
optional script. No framework, no build step, no npm, no analytics, no
tracking — by design. Open `web/index.html` in a browser and it works;
push the folder to any static host and it works there too.

## Pages

| file | purpose |
|---|---|
| `index.html` | Home — what NightRun is, screenshots, verified features, measured benchmarks |
| `architecture.html` | first-class engineering story: UEFI residency, memory model, pipeline, kernels, format, validation, tradeoffs |
| `install.html` | guided + manual installation for both targets, flashing safety, troubleshooting |
| `docs.html` | deep reference: commands, manifests, `.nrm`, tokenizer testing, benchmark methodology, limits, roadmap |

## Local preview

```sh
python3 -m http.server -d web 8080
# then open http://localhost:8080
```

Opening the files directly (`file://`) also works — all links and assets
are relative.

## GitHub Pages

Settings → Pages → deploy from branch, folder `/web` is not offered by
Pages directly, so either: (a) publish the `web/` folder to a `gh-pages`
branch (`git subtree push --prefix web origin gh-pages`), or (b) copy
`web/` to `/docs` and select the `/docs` folder option. Nothing else is
required — there is no build.

## Assets

- `assets/logo.svg` — the NIGHTRUN wordmark with the exact gradient from
  `crates/nr-gfx/src/theme.rs` (`LOGO_STOPS`).
- `assets/screenshots/*.png` — **real captures** taken with
  `cargo xtask run --img … --shot t:file.png` in QEMU, then
  palette-quantized with PIL (128 colors, no dither — the UI is
  flat-colored, so this is visually lossless at ~45% of the size).
  To refresh after UI changes: retake with the same command and
  re-quantize; keep total screenshot payload under ~150 KB.

## Keeping the site honest

Every number and command on these pages traces to a repository source:
benchmarks to `docs/benchmarks.md` / `docs/baseline-x86.md`, commands to
`README.md` / `CLAUDE.md` / `docs/installer.md`, Pi facts to
`docs/rpi5-uefi.md`. **When those files change, the site copy must be
re-checked** — the site never claims what the repo can't back. No
projections, no invented screenshots, no "up to" numbers.

## Why no framework

The site is a few documents. HTML+CSS handles it in ~2,000 lines total,
loads instantly, works without JavaScript (the script only adds copy
buttons and TOC highlighting), and will still build — because there is
nothing to build — in ten years.
