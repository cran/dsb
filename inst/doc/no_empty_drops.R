## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## -----------------------------------------------------------------------------
library(dsb)
# Specify isotype controls to use in step II 
isotypes = c("MouseIgG1kappaisotype_PROT", "MouseIgG2akappaisotype_PROT", 
             "Mouse IgG2bkIsotype_PROT", "RatIgG2bkIsotype_PROT")

# run ModelNegativeADTnorm to model ambient noise and implement step II
raw.adt.matrix = dsb::cells_citeseq_mtx
norm.adt = ModelNegativeADTnorm(cell_protein_matrix = raw.adt.matrix,
                                denoise.counts = TRUE,
                                use.isotype.control = TRUE,
                                isotype.control.name.vec = isotypes
                                )


## ----fig.width=7.5, fig.height=6----------------------------------------------
par(mfrow = c(2,2)); r = '#009ACD80'
lab = 'ModelNegativeADTnorm'
hist(norm.adt["CD4_PROT", ], breaks = 45, col = r, main = 'CD4', xlab = lab)
hist(norm.adt["CD8_PROT", ], breaks = 45, col = r, main = 'CD8', xlab = lab)
hist(norm.adt["CD19_PROT", ], breaks = 45, col = r, main = 'CD19', xlab = lab)
hist(norm.adt["CD18_PROT", ], breaks = 45, col = r, main = 'CD18', xlab = lab)

## ----eval = FALSE-------------------------------------------------------------
#  # these data were installed with the SeuratData package
#  # devtools::install_github('satijalab/seurat-data')
#  # library(SeuratData)
#  # InstallData(ds = 'bmcite') from https://doi.org/10.1016/j.cell.2019.05.031
#  # load bone marrow CITE-seq data
#  data('bmcite')
#  bm = bmcite; rm(bmcite)
#  
#  # Extract raw bone marrow ADT data
#  adt = GetAssayData(bm, slot = 'counts', assay = 'ADT')
#  
#  # unfortunately this data does not have isotype controls
#  dsb.norm = ModelNegativeADTnorm(cell_protein_matrix = adt,
#                                  denoise.counts = TRUE,
#                                  use.isotype.control = FALSE)

## ----eval = FALSE-------------------------------------------------------------
#  
#  # specify isotype controls
#  isotype.controls = c('isotype1', 'isotype 2')
#  # normalize ADTs
#  dsb.norm.2 = ModelNegativeADTnorm(cell_protein_matrix = adt,
#                                    denoise.counts = TRUE,
#                                    use.isotype.control = TRUE,
#                                    isotype.control.name.vec = isotype.controls
#                                    )

## ----fig.width=7, fig.height=3.5, eval = FALSE--------------------------------
#  library(ggplot2); theme_set(theme_bw())
#  plist = list(geom_vline(xintercept = 0, color = 'red'),
#               geom_hline(yintercept = 0, color = 'red'),
#               geom_point(size = 0.2, alpha = 0.1))
#  d = as.data.frame(t(dsb.norm))
#  
#  # plot distributions
#  p1 = ggplot(d, aes(x = CD4, y = CD8a)) + plist
#  p2 = ggplot(d, aes(x = CD19, y = CD3)) + plist
#  cowplot::plot_grid(p1,p2)
#  

## ----eval = FALSE-------------------------------------------------------------
#  bm = SetAssayData(bmcite, slot = 'data',
#                    assay = 'ADT',
#                    new.data = dsb.norm)
#  
#  # process RNA for WNN
#  DefaultAssay(bm) <- 'RNA'
#  bm <- NormalizeData(bm) %>%
#    FindVariableFeatures() %>%
#    ScaleData() %>%
#    RunPCA()
#  
#  # process ADT for WNN # see the main dsb vignette for an alternate version
#  DefaultAssay(bm) <- 'ADT'
#  VariableFeatures(bm) <- rownames(bm[["ADT"]])
#  bm = bm %>% ScaleData() %>% RunPCA(reduction.name = 'apca')
#  
#  # run WNN
#  bm <- FindMultiModalNeighbors(
#    bm, reduction.list = list("pca", "apca"),
#    dims.list = list(1:30, 1:18), modality.weight.name = "RNA.weight"
#  )
#  
#  bm <- FindClusters(bm, graph.name = "wsnn",
#                     algorithm = 3, resolution = 2,
#                     verbose = FALSE)
#  bm <- RunUMAP(bm, nn.name = "weighted.nn",
#                reduction.name = "wnn.umap",
#                reduction.key = "wnnUMAP_")
#  
#  p2 <- DimPlot(bm, reduction = 'wnn.umap',
#                group.by = 'celltype.l2',
#                label = TRUE, repel = TRUE,
#                label.size = 2.5) + NoLegend()
#  p2

