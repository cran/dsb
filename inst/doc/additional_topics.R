## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>", 
  eval = FALSE
)

## ---- eval=FALSE--------------------------------------------------------------
#  suppressMessages(library(SingleCellExperiment))
#  sce = SingleCellExperiment(assays = list(counts = cell.rna.raw), colData = cellmd)
#  # define the dsb normalized values as logcounts to use a common SingleCellExperiment / Bioconductor convention
#  adt = SummarizedExperiment(
#    assays = list(
#      'counts' = as.matrix(cell.adt.raw),
#      'logcounts' = as.matrix(cell.adt.dsb)
#      )
#    )
#  altExp(sce, "CITE") = adt

## ---- eval = FALSE------------------------------------------------------------
#  library(reticulate); sc = import("scanpy")
#  
#  # merge dsb-normalized protein and raw RNA data
#  combined_dat = rbind(cell.rna.raw, cell.adt.dsb)
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
#                                          empty_drop_matrix = empty_drop_citeseq_mtx,
#                                          # denoise with background mean only
#                                          denoise.counts = TRUE,
#                                          use.isotype.control = FALSE)
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
#  ## (QC the negative object to filter out cells with high RNA content)
#  # quick example below, different crteria can be used
#  # this step depends on dataset; see main vignette for more principled filtering
#  neg_object = subset(seurat_object, idents = "Negative", nGene < 80)
#  
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
#  # proceed with same tutorial workflow shown above.

## -----------------------------------------------------------------------------
#  library(dsb)
#  result.list =
#    DSBNormalizeProtein(
#      cell_protein_matrix = cells_citeseq_mtx[ ,1:50],
#      empty_drop_matrix = empty_drop_citeseq_mtx,
#      denoise.counts = TRUE,
#      use.isotype.control = TRUE,
#      isotype.control.name.vec = rownames(cells_citeseq_mtx)[67:70],
#      return.stats = TRUE
#    )
#  

## -----------------------------------------------------------------------------
#  names(result.list)

## -----------------------------------------------------------------------------
#  names(result.list$protein_stats)

## -----------------------------------------------------------------------------
#  result.list$technical_stats %>% head

## ---- eval = FALSE------------------------------------------------------------
#  dsb_norm_prot = DSBNormalizeProtein(
#                             cell_protein_matrix = cells_citeseq_mtx,
#                             empty_drop_matrix = empty_drop_citeseq_mtx,
#                             denoise.counts = TRUE,
#                             use.isotype.control = TRUE,
#                             isotype.control.name.vec = rownames(cells_citeseq_mtx)[67:70],
#                             # implement Quantile clipping
#                             quantile.clipping = TRUE
#                             # high and low otlier quantile across proteins to clip
#                             # the `quantile.clip` parameter can be adjusted:
#                             quantile.clip = c(0.001, 0.9995)
#                             )

## ---- eval = FALSE------------------------------------------------------------
#  
#  dsb_norm_prot = DSBNormalizeProtein(
#                             cell_protein_matrix = cells_citeseq_mtx,
#                             empty_drop_matrix = empty_drop_citeseq_mtx,
#                             denoise.counts = TRUE,
#                             use.isotype.control = TRUE,
#                             isotype.control.name.vec = rownames(cells_citeseq_mtx)[67:70],
#                             quantile.clipping = TRUE,
#                             scale.factor = 'mean.subtract'
#                             )
#  

## ---- eval = FALSE------------------------------------------------------------
#  # find outliers
#  pheatmap::pheatmap(apply(dsb_norm_prot, 1, function(x){
#    quantile(x,c(0.9999, 0.99, 0.98, 0.95, 0.0001, 0.01, 0.1))
#    }))
#  

## ---- eval=FALSE--------------------------------------------------------------
#  
#  dsb_object = DSBNormalizeProtein(cell_protein_matrix = dsb::cells_citeseq_mtx,
#                                   empty_drop_matrix = dsb::empty_drop_citeseq_mtx,
#                                   denoise.counts = TRUE,
#                                   isotype.control.name.vec = rownames(dsb::cells_citeseq_mtx)[67:70],
#                                   return.stats = TRUE)
#  d = as.data.frame(dsb_object$dsb_stats)
#  
#  # test correlation of background mean with the inferred dsb technical component
#  cor(d$cellwise_background_mean, d$dsb_technical_component)
#  
#  # test average isotype control value correlation with the background mean
#  isotype_names = rownames(dsb::cells_citeseq_mtx)[67:70]
#  cor(rowMeans(d[,isotype_names]), d$cellwise_background_mean)
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

