suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(name, default = NULL) {
  index <- match(name, args)
  if (is.na(index)) return(default)
  if (index == length(args)) stop(paste("Missing value for", name))
  args[[index + 1]]
}

task_id <- arg_value("--task-id")
work_dir <- normalizePath(arg_value("--workdir", "."), mustWork = TRUE)
data_dir <- normalizePath(
  arg_value("--data-dir", file.path(work_dir, "data", "processed")),
  mustWork = TRUE
)
if (is.null(task_id)) stop("--task-id is required")

tasks_path <- arg_value("--tasks")
protocol_path <- arg_value("--protocol")
if (is.null(tasks_path) || is.null(protocol_path)) stop("--tasks and --protocol are required")
tasks_payload <- fromJSON(tasks_path, simplifyVector = FALSE)
protocol <- fromJSON(protocol_path, simplifyVector = FALSE)
task_index <- which(vapply(tasks_payload$tasks, function(x) identical(x$task_id, task_id), logical(1)))
if (length(task_index) != 1) stop(paste("Unknown or duplicate task", task_id))
task <- tasks_payload$tasks[[task_index]]
source_info <- protocol$scenario_sources[[task$scenario]]

cell_type_column <- switch(
  task$scenario,
  TEA_s1 = "celltype.l2",
  TEA_s2 = "celltype.l2",
  BMMC_s1 = "celltype.l1",
  BMMC_s2 = "celltype.l1",
  Retina = "cell_type__custom",
  `Ab-seq` = "celltype.l2",
  stop(paste("Unsupported scenario", task$scenario))
)

metadata <- list()
references <- list()
for (observation in task$observations) {
  source_dir <- file.path(data_dir, task$data_root, observation$source_subdir)
  meta_path <- file.path(source_dir, "meta.rds")
  if (!file.exists(meta_path)) stop(paste("Missing metadata", meta_path))
  meta <- readRDS(meta_path)
  if (!cell_type_column %in% colnames(meta)) {
    stop(paste("Missing cell type column", cell_type_column, "in", meta_path))
  }
  source_cell_id <- rownames(meta)
  if (is.null(source_cell_id) || any(!nzchar(source_cell_id))) {
    stop(paste("Metadata requires row names", meta_path))
  }
  instance <- observation$instance_batch
  cell_id <- paste(source_cell_id, instance, sep = "::")
  observed <- unlist(observation$observed_modalities)
  composition <- paste(toupper(observed), collapse = "+")
  metadata[[length(metadata) + 1]] <- data.frame(
    cell_id = cell_id,
    source_cell_id = source_cell_id,
    instance_batch = instance,
    base_batch = observation$base_batch,
    cell_type = as.character(meta[[cell_type_column]]),
    modality = composition,
    identity_policy = observation$identity_policy,
    stringsAsFactors = FALSE,
    row.names = cell_id
  )
  modality_paths <- list()
  for (modality in observed) {
    rds_path <- file.path(source_dir, paste0(modality, ".rds"))
    if (file.exists(rds_path)) {
      modality_paths[[modality]] <- normalizePath(rds_path, winslash = "/")
    } else if (identical(modality, "atac")) {
      matrix_path <- file.path(source_dir, "atac.mtx")
      if (!file.exists(matrix_path)) stop(paste("Missing modality matrix", rds_path))
      modality_paths[[modality]] <- list(
        matrix = normalizePath(matrix_path, winslash = "/"),
        features = normalizePath(file.path(source_dir, "peak.csv"), winslash = "/"),
        barcodes = normalizePath(file.path(source_dir, "bcd.csv"), winslash = "/")
      )
    } else {
      stop(paste("Missing modality matrix", rds_path))
    }
  }
  references[[length(references) + 1]] <- list(
    instance_batch = instance,
    base_batch = observation$base_batch,
    source_subdir = observation$source_subdir,
    identity_policy = observation$identity_policy,
    cell_id_suffix = paste0("::", instance),
    modalities = modality_paths
  )
}

metadata <- do.call(rbind, metadata)
if (anyDuplicated(metadata$cell_id)) stop("Task contains duplicate cell_id values")
output_dir <- file.path(work_dir, "data", "tasks", task_id)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(metadata, file.path(output_dir, "metadata_evaluation.csv"), row.names = FALSE, quote = TRUE)
training <- metadata[, c("cell_id", "instance_batch", "base_batch", "modality", "identity_policy")]
write.csv(training, file.path(output_dir, "metadata_training.csv"), row.names = FALSE, quote = TRUE)
write_json(
  list(
    schema_version = "1.0",
    task_id = task_id,
    scenario = task$scenario,
    replicate = task$replicate,
    cell_count = nrow(metadata),
    hidden_identity_available_to_training = FALSE,
    cell_type_available_to_training = FALSE,
    observations = references
  ),
  file.path(output_dir, "data_references.json"),
  auto_unbox = TRUE,
  pretty = TRUE
)
cat(sprintf("Prepared %s with %d cell instances\n", task_id, nrow(metadata)))
