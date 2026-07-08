# Micro-C preprocessing

`touche` does not run raw FASTQ alignment or cooler generation. Use an external
workflow such as [distiller-nf](https://github.com/open2c/distiller-nf) for raw
Micro-C processing, then use `touche preprocess` to convert, filter, QC, and
cache the resulting pairs files.

## External distiller-nf step

[Danko-Lab/E-P_contacts](https://github.com/Danko-Lab/E-P_contacts) stores
example distiller-nf configuration files under
[`Micro-C_basic_processing/`](https://github.com/Danko-Lab/E-P_contacts/tree/main/Micro-C_basic_processing).

Those YAML files were used to map raw Micro-C data and generate `.cool`/`.mcool`
files. For downstream `touche` analyses, the important output is a pairs file
with mapq columns available.

When preparing your own data with distiller-nf, match the reference assumptions:

```yaml
parsing_options: '--add-columns mapq'
drop_readid: True
```

The downstream scripts in the reference workflow expect cis pairs with both
sides passing a mapq threshold, usually 30.

## Analysis-ready pairs

The reference workflow documents this filtering shape:

```bash
zcat prefix.rep1.pairs.gz prefix.rep2.pairs.gz prefix.rep3.pairs.gz \
  | awk 'BEGIN {OFS = "\t"} ; {if ($1 == "." && $2 == $4 && $9 >= 30 && $10 >= 30) {print $2, $3, $4, $5, $6, $7, $8, $9, $10}}' \
  > prefix.nodups_30_intra.pairs
```

In `touche`, use:

```bash
uv run touche preprocess filter-pairs \
  --pairs prefix.pairs.gz \
  --out prefix.nodups_30_intra.pairs.gz \
  --min-mapq 30 \
  --cis-only
```

By default, `filter-pairs` writes the canonical 9-column `touche` format:

```text
chrA  posA  chrB  posB  strandA  strandB  read_type  mapqA  mapqB
```

This corresponds to the reference filter after dropping the read ID column.

## Conversion without filtering

To convert a distiller/pairtools-style file into the canonical `touche` format
without filtering:

```bash
uv run touche preprocess convert-pairs \
  --pairs prefix.pairs.gz \
  --from distiller \
  --to touche \
  --out prefix.touche.pairs.gz
```

## QC summary

Write a compact QC JSON file:

```bash
uv run touche preprocess qc \
  --pairs prefix.nodups_30_intra.pairs.gz \
  --source touche \
  --out prefix.qc.json
```

The QC summary includes total parsed rows, cis/trans counts, mapq-pass/fail
counts, per-chromosome counts, and a coarse cis-distance histogram.

## NPZ cache

For repeated downstream analyses, build chromosome-sharded NPZ caches:

```bash
uv run touche preprocess build-cache \
  --pairs prefix.nodups_30_intra.pairs.gz \
  --source touche \
  --cache-dir .cache/touche/prefix \
  --prefix prefix
```

The cache is intentionally sharded by chromosome so downstream commands do not
need to unpack a monolithic whole-genome archive.

## Boundary

Use distiller-nf or an equivalent external workflow for:

- alignment
- parsing raw reads into pairs
- duplicate marking/removal
- cooler/mcool generation

Use `touche preprocess` for:

- pairs format conversion
- cis/mapq filtering
- analysis-ready QC
- chromosome-sharded numeric caches
