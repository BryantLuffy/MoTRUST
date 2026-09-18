suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
index_path <- normalizePath(args[[1]], mustWork=TRUE)
index <- fromJSON(index_path, simplifyVector=FALSE)
idx <- index$scenarios
dest <- args[[2]]
root <- if (length(args) >= 3) args[[3]] else getwd()
base <- if (!is.null(index$path_base) && index$path_base == 'index') dirname(index_path) else root
resolve_reference <- function(p) {
  if (grepl('^([A-Za-z]:[/\\]|/|\\\\)', p)) p else file.path(base, p)
}
for (scenario in c('TEA_s1','BMMC_s1','Retina')) {
  for (batch in names(idx[[scenario]])) {
    item <- idx[[scenario]][[batch]]
    if (!all(c('rna','atac') %in% names(item$modalities))) next
    path <- resolve_reference(item$metadata)
    if (grepl('\\.csv$', path, ignore.case=TRUE)) {
      x <- read.csv(path, check.names=FALSE, stringsAsFactors=FALSE)
      if (!'original_barcode' %in% names(x)) stop('CSV metadata requires original_barcode')
    } else {
      x <- as.data.frame(readRDS(path))
      x$original_barcode <- rownames(x)
    }
    folder <- file.path(dest, scenario)
    dir.create(folder, recursive=TRUE, showWarnings=FALSE)
    write.csv(x, file.path(folder,paste0(batch,'.csv')),row.names=FALSE)
  }
}
