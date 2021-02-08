## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.path = "man/figures/README-",
  out.width = "100%"
)

## ----eval = FALSE-------------------------------------------------------------
#  # install dsb package
#  library(dsb)
#  
#  adt_norm = DSBNormalizeProtein(
#    # remove ambient protien noise reflected in counts from empty droplets
#    cell_protein_matrix = cells_citeseq_mtx, # cell-containing droplet raw protein count matrix
#    empty_drop_matrix = empty_drop_citeseq_mtx, # empty/background droplet raw protein counts
#  
#    # recommended step II: model and remove the technical component of each cell's protein library
#    denoise.counts = TRUE, # model and remove each cell's technical component
#    use.isotype.control = TRUE, # use isotype controls to define the technical component
#    isotype.control.name.vec = rownames(cells_citeseq_mtx)[67:70] # vector of isotype control names
#    )

## ---- eval = FALSE------------------------------------------------------------
#  library(dsb)
#  library(Seurat) # Seurat Version 3 used below for the Read10X function
#  library(tidyverse) # used for ggplot and %>% (tidyverse is not a dsb dependency)
#  
#  # read raw data using the Seurat function "Read10X"
#  raw = Read10X("data/10x_data/10x_pbmc5k_V3/raw_feature_bc_matrix/")
#  
#  # Read10X formats the output as a list for the ADT and RNA assays: split this list
#  prot = raw$`Antibody Capture`
#  rna = raw$`Gene Expression`
#  
#  # create a metadata dataframe of simple qc stats for each droplet
#  rna_size = log10(Matrix::colSums(rna))
#  prot_size = log10(Matrix::colSums(prot))
#  ngene = Matrix::colSums(rna > 0)
#  mtgene = grep(pattern = "^MT-", rownames(rna), value = TRUE)
#  propmt = Matrix::colSums(rna[mtgene, ]) / Matrix::colSums(rna)
#  md = as.data.frame(cbind(propmt, rna_size, ngene, prot_size))
#  md$bc = rownames(md)

## ---- eval = FALSE------------------------------------------------------------
#  p1 = ggplot(md[md$rna_size > 0, ], aes(x = rna_size)) + geom_histogram(fill = "dodgerblue") + ggtitle("RNA library size \n distribution")
#  p2 = ggplot(md[md$prot_size> 0, ], aes(x = prot_size)) + geom_density(fill = "firebrick2") + ggtitle("Protein library size \n distribution")
#  cowplot::plot_grid(p1, p2, nrow = 1)

## ---- eval = FALSE------------------------------------------------------------
#  # define a vector of background / empty droplet barcodes based on protein library size and mRNA content
#  background_drops = md[md$prot_size > 1.4 & md$prot_size < 2.5 & md$ngene < 80, ]$bc
#  negative_mtx_rawprot = prot[ , background_drops] %>% as.matrix()
#  
#  # define a vector of cell-containing droplet barcodes based on protein library size and mRNA content
#  positive_cells = md[md$prot_size > 2.8 & md$ngene < 3000 & md$ngene > 200 & propmt <0.2, ]$bc
#  cells_mtx_rawprot = prot[ , positive_cells] %>% as.matrix()
#  

## ---- eval = FALSE------------------------------------------------------------
#  length(positive_cells)

## ---- eval = FALSE------------------------------------------------------------
#  #normalize protein data for the cell containing droplets with the dsb method.
#  dsb_norm_prot = DSBNormalizeProtein(
#    cell_protein_matrix = cells_mtx_rawprot, # cell containing droplets
#    empty_drop_matrix = negative_mtx_rawprot, # estimate ambient noise with the background drops
#    denoise.counts = TRUE, # model and remove each cell's technical component
#    use.isotype.control = TRUE, # use isotype controls to define the technical component
#    isotype.control.name.vec = rownames(cells_mtx_rawprot)[30:32] # names of isotype control abs
#    )

## ---- eval = FALSE------------------------------------------------------------
#  # filter raw protein, RNA and metadata to only include cell-containing droplets
#  cells_rna = rna[ ,positive_cells]
#  md = md[positive_cells, ]
#  
#  # create Seurat object with cell-containing drops (min.cells is a gene filter, not a cell filter)
#  s = Seurat::CreateSeuratObject(counts = cells_rna, meta.data = md, assay = "RNA", min.cells = 20)
#  
#  # add DSB normalized "dsb_norm_prot" protein data to a assay called "CITE" created in step II
#  s[["CITE"]] = Seurat::CreateAssayObject(data = dsb_norm_prot)

## ---- eval = FALSE------------------------------------------------------------
#  # define Euclidean distance matrix on dsb normalized protein data (without isotype controls)
#  dsb = s@assays$CITE@data[1:29, ]
#  p_dist = dist(t(dsb))
#  p_dist = as.matrix(p_dist)
#  
#  # Cluster using Seurat
#  s[["p_dist"]] = Seurat::FindNeighbors(p_dist)$snn
#  s = Seurat::FindClusters(s, resolution = 0.5, graph.name = "p_dist")

## ---- eval = FALSE------------------------------------------------------------
#  # calculate the average of each protein separately for each cluster
#  prots = rownames(s@assays$CITE@data)
#  adt_plot = adt_data %>%
#    group_by(seurat_clusters) %>%
#    summarize_at(.vars = prots, .funs = mean) %>%
#    column_to_rownames("seurat_clusters")
#  # plot a heatmap of the average dsb normalized values for each cluster
#  pheatmap::pheatmap(t(adt_plot), color = viridis::viridis(25, option = "B"), fontsize_row = 8, border_color = NA)
#  

## ---- eval = FALSE------------------------------------------------------------
#  library(reticulate); use_virtualenv("r-reticulate")
#  library(umap)
#  
#  # set umap config
#  config = umap.defaults
#  config$n_neighbors = 40
#  config$min_dist = 0.4
#  
#  # run umap directly on dsb normalized values
#  ump = umap(t(s2_adt3), config = config)
#  umap_res = as.data.frame(ump$layout)
#  colnames(umap_res) = c("UMAP_1", "UMAP_2")
#  
#  # save results dataframe
#  df_dsb = cbind(s@meta.data, umap_res, as.data.frame(t(s@assay$CITE@data)))
#  
#  # visualizatons below were made directly from the data frame df_dsb above with ggplot

## ---- eval=FALSE--------------------------------------------------------------
#  library(SingleCellExperiment)
#  sce = SingleCellExperiment(assays = list(counts = cells_rna), colData = md)
#  dsb_adt = SummarizedExperiment(as.matrix(count_prot))
#  altExp(sce, "CITE") = dsb_adt
#  logcounts(altExp(sce)) = dsb_norm_prot

## ---- eval = FALSE------------------------------------------------------------
#  library(reticulate); sc = import("scanpy")
#  
#  # merge dsb-normalized protein and raw RNA data
#  combined_dat = rbind(count_rna, dsb_norm_prot)
#  s[["combined_data"]] = CreateAssayObject(data = combined_dat)
#  
#  # create Anndata Object
#  adata_seurat = sc$AnnData(
#      X   = t(GetAssayData(s,assay = "combined_data")),
#      obs = seurat@meta.data,
#      var = GetAssay(seurat)[[]]
#      )

## ---- eval = FALSE------------------------------------------------------------
#  # suggested workflow if isotype controls are not included
#  dsb_rescaled = DSBNormalizeProtein(cell_protein_matrix = cells_citeseq_mtx,
#                                     empty_drop_matrix = empty_drop_citeseq_mtx,
#                                     # do not denoise each cell's technical component
#                                     denoise.counts = FALSE)

## ---- eval = FALSE------------------------------------------------------------
#  
#  dsb_rescaled = dsb::DSBNormalizeProtein(cell_protein_matrix = cells_citeseq_mtx,
#                                     empty_drop_matrix = empty_drop_citeseq_mtx,
#                                     # denoise with background mean only
#                                     denoise.counts = TRUE,
#                                     use.isotype.control = FALSE)
#  

## ---- eval=FALSE--------------------------------------------------------------
#  # raw = Read10X see above -- path to cell ranger outs/raw_feature_bc_matrix ;
#  
#  # partial thresholding to slightly subset negative drops include all with 5 unique mRNAs
#  seurat_object = CreateSeuratObject(raw, min.genes = 5)
#  
#  # demultiplex (positive.quantile can be tuned to dataset depending on size)
#  seurat_object = HTODemux(seurat_object, assay = "HTO", positive.quantile = 0.99)
#  Idents(seurat_object) = "HTO_classification.global"
#  
#  # subset empty drop/background and cells
#  neg_object = subset(seurat_object, idents = "Negative")
#  singlet_object = subset(seurat_object, idents = "Singlet")
#  
#  # non sparse CITEseq data store more efficiently in a regular matrix
#  neg_adt_matrix = GetAssayData(neg_object, assay = "CITE", slot = 'counts') %>% as.matrix()
#  positive_adt_matrix = GetAssayData(singlet_object, assay = "CITE", slot = 'counts') %>% as.matrix()
#  
#  # normalize the data with dsb
#  dsb_norm_prot = DSBNormalizeProtein(
#                             cell_protein_matrix = cells_mtx_rawprot,
#                             empty_drop_matrix = negative_mtx_rawprot,
#                             denoise.counts = TRUE,
#                             use.isotype.control = TRUE,
#                             isotype.control.name.vec = rownames(cells_mtx_rawprot)[30:32])
#  
#  # now add the normalized dat back to the object (the singlets defined above as "object")
#  singlet_object[["CITE"]] = CreateAssayObject(data = dsb_norm_prot)
#  

## ---- eval = FALSE------------------------------------------------------------
#  library(Seurat) # for Read10X helper function
#  
#  # path_to_reads = here("data/")
#  umi.files = list.files(path_to_reads, full.names=T, pattern = "10x" )
#  umi.list = lapply(umi.files, function(x) Read10X(data.dir = paste0(x,"/outs/raw_feature_bc_matrix/")))
#  prot = rna = list()
#  for (i in 1:length(umi.list)) {
#    prot[[i]] = umi.list[[i]]`Antibody Capture`
#    rna[[i]] = umi.list[[i]]`Gene Expression`
#    colnames(prot[[i]]) = paste0(colnames(prot[[i]]),"_", i )
#    colnames(rna[[i]]) = paste0(colnames(rna[[i]]),"_", i )
#  }
#  prot = do.call(cbind, prot)
#  rna = do.call(cbind, rna)
#  # proceed with step 1 in tutorial - define background and cell containing drops for dsb
#  

## ---- eval=FALSE--------------------------------------------------------------
#  # at step 3
#  pv = md[positive_cells, ]; pv$class = "cell_containing"
#  nv = md[background_drops, ]; nv$class = "background"
#  ddf = rbind(pv, nv)
#  # plot
#  p = ggplot(ddf, aes(x = prot_size, fill = class, color = class )) +
#    theme_bw() +
#    geom_histogram(aes(y=..count..), alpha=0.5, bins = 50,position="identity")+
#    ggtitle(paste0("theoretical max barcodes = ", nrow(md),
#                   "\n", "cell containing drops after QC = ", nrow(pv),
#                   "\n", "negative droplets = ", nrow(nv))) +
#    theme(legend.position = c(0.8, 0.7))
#  # add a marginal histogram
#  xtop = cowplot::axis_canvas(p, axis = "x") +
#    geom_density(data = ddf, aes(x = prot_size, fill = class), alpha = 0.5)
#  # merge plots
#  p2 = cowplot::ggdraw(cowplot::insert_xaxis_grob(p, xtop, grid::unit(.4, "null"), position = "top"))
#  

## ---- eval=FALSE--------------------------------------------------------------
#  # step 1: confirm low correlation between µ1 and µ2 from the Gaussian mixture.
#  adtu_log1 = log(empty_drop_citeseq_mtx + 10)
#  adt_log1 = log(cells_citeseq_mtx + 10)
#  
#  # rescale
#  mu_u1 = apply(adtu_log1, 1 , mean)
#  sd_u1 = apply(adtu_log1, 1 , sd)
#  norm_adt = apply(adt_log1, 2, function(x) (x  - mu_u1) / sd_u1)
#  
#  # run per-cellgaussian mixture
#  library(mclust)
#  cm = apply(norm_adt, 2, function(x) {
#  			g = Mclust(x, G=2, warn = TRUE, verbose = FALSE)
#  			return(g)
#  		})
#  # tidy model fit data
#  mu1 = lapply(cm, function(x){ x$parameters$mean[1] }) %>% base::unlist(use.names = FALSE)
#  mu2 = lapply(cm, function(x){ x$parameters$mean[2] }) %>% base::unlist(use.names = FALSE)
#  # test correlation
#  cor.test(mu1, mu2)

