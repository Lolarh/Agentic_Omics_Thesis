getwd()
setwd("/Users/lola/Desktop/Agentic_Thesis")
getwd()
list.files()
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("scRNAseq")

library(scRNAseq)

sce <- MuraroPancreasData()

print(sce)

table(sce$label, useNA = "ifany")

head(colData(sce))

dir.create(
  "data/muraro",
  recursive = TRUE,
  showWarnings = FALSE
)

# Export the file
metadata <- data.frame(
  cell_id = colnames(sce),
  label = sce$label)

write.csv(
  metadata,
  "data/muraro/muraro_cell_annotations.csv",
  row.names = FALSE)

file.exists("data/muraro/muraro_cell_annotations.csv")

counts_muraro <- assay(sce, "counts")

print(counts_muraro[1:5, 1:5])

print(class(counts_muraro))

print(range(counts_muraro))

print(mean(counts_muraro %% 1 == 0))
