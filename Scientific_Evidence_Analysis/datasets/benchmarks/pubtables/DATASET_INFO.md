# PubTables-1M

## Purpose and official sources

PubTables-1M is a table extraction benchmark for table detection, table structure
recognition, and functional analysis.

- Hugging Face dataset: [bsmock/pubtables-1m](https://huggingface.co/datasets/bsmock/pubtables-1m)
- Official implementation and data guide:
  [microsoft/table-transformer](https://github.com/microsoft/table-transformer)
- Dataset paper:
  [PubTables-1M: Towards comprehensive table extraction from unstructured documents](https://openaccess.thecvf.com/content/CVPR2022/html/Smock_PubTables-1M_Towards_Comprehensive_Table_Extraction_From_Unstructured_Documents_CVPR_2022_paper.html)
- Full release: [Microsoft Research Open Data](https://msropendata.com/datasets/505fcbe3-1383-42b1-913a-f651b8b712d3)

The official Table Transformer README reports 575,305 annotated document pages
for table detection and 947,642 fully annotated tables for structure recognition
and functional analysis. The Hugging Face dataset card says it currently supports
download as 18 `tar.gz` files and that Hugging Face Datasets integration is not
implemented. The Hugging Face dataset viewer exposes no usable splits or Parquet
files for this repository.

## Download status and subset size

**Downloaded PubTables-1M records in this project: 0.**

No PubTables data archive has been downloaded. The official files are split-level
compressed archives, not independently addressable table examples. Structure
annotations are in large split archives (the README lists 93,834 test, 758,849
train, and 94,959 validation XML annotation files), while matching images reside
in separate image archives. The source documents do not provide an official
per-table sample endpoint or a supported streaming/subsetting API. Extracting a
valid small table/image sample would therefore require downloading and traversing
large archives, which conflicts with the requested resource restriction. We
deliberately did not download all archives or any bulk image archive.

The downloader records this status at `data/subset_status.json`. If an official
sample endpoint or a user-provided licensed archive subset becomes available,
the local inspection workflow can be extended without treating unpaired
annotations as a usable sample.

## Original data formats and annotations

The official repository describes 18 archives:

- Structure recognition: filelists, train/validation/test XML annotations,
  train/validation/test cropped-table images, and table-word annotations.
- Detection: filelists, train/validation/test XML annotations, page-image
  archives, and page-word annotations.
- PDF annotations: detailed table annotations in PDF coordinates.

Structure XML uses PASCAL VOC-style boxes. The complete dataset includes table,
row, column, cell, header, and projected-row-header structure, including blank
cells. Depending on the archive, annotations can include bounding boxes, text,
and table/cell structure; source provenance and identifiers are associated with
archive filenames and filelists. Exact per-file field contents should be checked
against the official archive/readme before use.

The Hugging Face dataset card does not provide a row-level `load_dataset()`
interface. A dataset viewer failure to infer splits is not evidence that the data
is absent; the official data card explicitly documents tar.gz delivery instead.

## Licence and use conditions

The official Hugging Face dataset card declares **CDLA-Permissive-2.0**
([licence](https://cdla.io/permissive-2-0/)). The dataset license applies to
data, not the software license of the separate Table Transformer code repository.
Review the licence and attribution obligations before redistribution or release
of derived data. Source-paper content and associated third-party materials may
have additional rights; the project README does not establish a separate rights
clearance for every source document.

## Relevance to this research

If a suitable, correctly paired subset is obtained, PubTables-1M can help evaluate
table-region and table-structure evidence extraction, including row/column/cell
association, header recognition, bounding-box provenance, and evidence contained
inside scientific tables. It could complement text retrieval and claim analysis
by measuring whether relevant tabular results are located and represented
correctly.

PubTables-1M is a table extraction benchmark, not a flood-susceptibility dataset
or a question-answering benchmark. It cannot itself validate flood science or
claim-contradiction reasoning.

## Limitations

- The full data scale is too large for this laptop workflow; no PubTables subset
  has been downloaded.
- No local sample counts, split distribution, annotations, or image dimensions
  are claimed.
- Separate annotation and image archives make it unsafe to use partial
  annotations without a verified matching image.
- A future subset should record archive version, original split, table/image IDs,
  licence/provenance, and matching checksums.
- No model training or GPU use has been performed.
