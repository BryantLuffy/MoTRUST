suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(Matrix))

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(name, default = NULL) {
  index <- match(name, args)
  if (is.na(index)) return(default)
  if (index == length(args)) stop(paste("Missing value for", name))
  args[[index + 1]]
}

work_dir <- normalizePath(arg_value("--workdir", "."), mustWork = TRUE)
data_dir <- normalizePath(arg_value("--data-dir", file.path(work_dir, "data", "processed")), mustWork = TRUE)
only_scenario <- arg_value("--scenario", NULL)
protocol_path <- arg_value("--protocol")
if (is.null(protocol_path)) stop("--protocol is required")
protocol <- fromJSON(protocol_path, simplifyVector = FALSE)
cache_root <- file.path(work_dir, "data", "cache")
dir.create(cache_root, recursive = TRUE, showWarnings = FALSE)

write_lines <- function(values, path) {
  writeLines(enc2utf8(as.character(values)), con = path, useBytes = TRUE)
}

index <- list()
for (scenario in protocol$scenario_order) {
  if (!is.null(only_scenario) && !identical(scenario, only_scenario)) next
  info <- protocol$scenario_sources[[scenario]]
  cache_group <- switch(
    scenario,
    TEA_s2 = "TEA_s1",
    BMMC_s2 = "BMMC_s1",
    scenario
  )
  scenario_index <- list()
  for (base_batch in names(info$base_batches)) {
    source_subdir <- info$base_batches[[base_batch]]
    source_dir <- file.path(data_dir, info$data_root, source_subdir)
    cache_dir <- file.path(cache_root, cache_group, gsub("[\\/]", "__", source_subdir))
    dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
    modality_index <- list()
    for (modality in unlist(info$modalities)) {
      rds_path <- file.path(source_dir, paste0(modality, ".rds"))
      original_mtx <- file.path(source_dir, paste0(modality, ".mtx"))
      if (file.exists(rds_path)) {
        output_mtx <- file.path(cache_dir, paste0(modality, ".mtx"))
        feature_path <- file.path(cache_dir, paste0(modality, "_features.tsv"))
        barcode_path <- file.path(cache_dir, paste0(modality, "_barcodes.tsv"))
        if (!file.exists(output_mtx) || !file.exists(feature_path) || !file.exists(barcode_path)) {
          cat(sprintf("Converting %s %s\n", scenario, paste(base_batch, modality)))
          object <- readRDS(rds_path)
          dense <- data.matrix(object)
          sparse <- Matrix(dense, sparse = TRUE)
          rm(object, dense)
          invisible(gc())
          writeMM(sparse, output_mtx)
          write_lines(rownames(sparse), feature_path)
          write_lines(colnames(sparse), barcode_path)
          rm(sparse)
          invisible(gc())
        }
        modality_index[[modality]] <- list(
          matrix = normalizePath(output_mtx, winslash = "/"),
          features = normalizePath(feature_path, winslash = "/"),
          barcodes = normalizePath(barcode_path, winslash = "/"),
          orientation = "features_by_cells",
          source = normalizePath(rds_path, winslash = "/")
        )
      } else if (file.exists(original_mtx)) {
        feature_path <- file.path(source_dir, if (identical(modality, "atac")) "peak.csv" else paste0(modality, "_features.csv"))
        barcode_path <- file.path(source_dir, if (identical(modality, "atac")) "bcd.csv" else paste0(modality, "_barcodes.csv"))
        if (!file.exists(feature_path) || !file.exists(barcode_path)) stop(paste("Missing names for", original_mtx))
        modality_index[[modality]] <- list(
          matrix = normalizePath(original_mtx, winslash = "/"),
          features = normalizePath(feature_path, winslash = "/"),
          barcodes = normalizePath(barcode_path, winslash = "/"),
          orientation = "features_by_cells",
          source = normalizePath(original_mtx, winslash = "/")
        )
      }
    }
    scenario_index[[base_batch]] <- list(
      source_subdir = source_subdir,
      metadata = normalizePath(file.path(source_dir, "meta.rds"), winslash = "/"),
      modalities = modality_index
    )
  }
  index[[scenario]] <- scenario_index
}

index_path <- file.path(cache_root, if (is.null(only_scenario)) "cache_index.json" else paste0("cache_index_", only_scenario, ".json"))
write_json(
  list(schema_version = "1.0", orientation = "features_by_cells", scenarios = index),
  index_path,
  auto_unbox = TRUE,
  pretty = TRUE
)
cat(sprintf("Cache index: %s\n", index_path))
