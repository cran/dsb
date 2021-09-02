## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.path = "man/figures/README-",
  out.width = "100%"
)

## ----eval = FALSE-------------------------------------------------------------
#  install.packages('dsb')

## ---- eval = FALSE------------------------------------------------------------
#  library(dsb)
#  adt_norm = DSBNormalizeProtein(
#    # cell-containing droplet raw protein count matrix
#    cell_protein_matrix = cells_citeseq_mtx,
#    # empty/background droplet raw protein counts
#    empty_drop_matrix = empty_drop_citeseq_mtx,
#  
#    # recommended step: model + remove the technical component of each cell's protein library
#    denoise.counts = TRUE,
#    use.isotype.control = TRUE,
#    isotype.control.name.vec = rownames(cells_citeseq_mtx)[67:70]
#    )
#  

## ---- eval = FALSE------------------------------------------------------------
#  # load packages used in this vignette
#  suppressMessages(library(dsb))
#  suppressMessages(library(Seurat))
#  suppressMessages(library(tidyverse))
#  
#  # read raw data using the Seurat function "Read10X"
#  raw = Seurat::Read10X("data/raw_feature_bc_matrix/")
#  cells = Seurat::Read10X("data/filtered_feature_bc_matrix/")
#  
#  # define a vector of cell-containing barcodes and remove them from unfiltered data
#  stained_cells = colnames(cells$`Gene Expression`)
#  background = setdiff(colnames(raw$`Gene Expression`), stained_cells)
#  
#  # split the data into separate matrices per assay
#  prot = raw$`Antibody Capture`
#  rna = raw$`Gene Expression`
#  
#  # create metadata of droplet QC stats used in standard scRNAseq processing
#  rna_size = log10(Matrix::colSums(rna))
#  prot_size = log10(Matrix::colSums(prot))
#  ngene = Matrix::colSums(rna > 0)
#  mtgene = grep(pattern = "^MT-", rownames(rna), value = TRUE)
#  propmt = Matrix::colSums(rna[mtgene, ]) / Matrix::colSums(rna)
#  md = as.data.frame(cbind(propmt, rna_size, ngene, prot_size))
#  md$bc = rownames(md)
#  md$droplet_class = ifelse(test = md$bc %in% stained_cells, yes = 'cell', no = 'background')
#  
#  # filter barcodes to only include those with data for both assays
#  md = md %>% dplyr::filter(rna_size > 0 & prot_size > 0 )
#  

## ---- eval = FALSE------------------------------------------------------------
#  ggplot(md, aes(x = log10(ngene), y = prot_size )) +
#    theme_bw() +
#    geom_bin2d(bins = 300) +
#    scale_fill_viridis_c(option = "C") +
#    facet_wrap(~droplet_class)

## ---- eval = FALSE------------------------------------------------------------
#  
#  cellmd = md %>% filter(droplet_class == 'cell')
#  plot_aes = list(theme_bw(), geom_point(shape = 21 , stroke = 0, size = 0.7), scale_fill_viridis_c(option = "C"))
#  p1 = ggplot(cellmd, aes(x = rna_size )) + geom_histogram(bins = 50) + theme_bw() + xlab("log10 RNA library size")
#  p2 = ggplot(cellmd, aes(x = propmt)) + geom_histogram(bins = 50) + theme_bw() + xlab("mitochondrial read proportion")
#  p3 = ggplot(cellmd, aes(x = log10(ngene), y = rna_size, fill = propmt )) + plot_aes
#  p4 = ggplot(cellmd, aes(x = ngene, y = prot_size, fill = propmt )) + plot_aes
#  p1+p2+p3+p4
#  

## ---- eval = FALSE------------------------------------------------------------
#  # calculate statistical thresholds for droplet filtering.
#  rna_size_min = median(cellmd$rna_size) - (3*mad(cellmd$rna_size))
#  rna_size_max = median(cellmd$rna_size) + (3*mad(cellmd$rna_size))
#  prot_size_min = median(cellmd$prot_size) - (3*mad(cellmd$prot_size))
#  prot_size_max = median(cellmd$prot_size) + (3*mad(cellmd$prot_size))
#  
#  # filter rows based on droplet qualty control metrics
#  positive_cells = cellmd[
#      cellmd$prot_size > prot_size_min &
#      cellmd$prot_size < prot_size_max &
#      cellmd$propmt < 0.14 &
#      cellmd$rna_size > rna_size_min &
#      cellmd$rna_size < rna_size_max, ]$bc
#  cells_mtx_rawprot = as.matrix(prot[ , positive_cells])

## ---- eval = FALSE------------------------------------------------------------
#  length(positive_cells)

## ---- eval = FALSE------------------------------------------------------------
#  # define a vector of background droplet barcodes based on protein library size and mRNA content
#  background_drops = md[md$prot_size > 1.5 & md$prot_size < 3 & md$ngene < 100, ]$bc
#  negative_mtx_rawprot = as.matrix(prot[ , background_drops])
#  

## ---- eval = FALSE------------------------------------------------------------
#  # calculate quantiles of the raw protein matrix
#  d1 = data.frame(pmax = apply(cells_mtx_rawprot, 1, max)) %>%
#    rownames_to_column('prot') %>% arrange(pmax) %>% head()

## ---- eval = FALSE------------------------------------------------------------
#  # remove non staining CD34 protein
#  prot_names = rownames(cells_mtx_rawprot)
#  cells_mtx_rawprot = cells_mtx_rawprot[!prot_names == 'CD34_TotalSeqB', ]
#  negative_mtx_rawprot = negative_mtx_rawprot[!prot_names == 'CD34_TotalSeqB', ]

## ---- eval = FALSE------------------------------------------------------------
#  #normalize protein data for the cell containing droplets with the dsb method.
#  dsb_norm_prot = DSBNormalizeProtein(
#    cell_protein_matrix = cells_mtx_rawprot,
#    empty_drop_matrix = negative_mtx_rawprot,
#    denoise.counts = TRUE,
#    use.isotype.control = TRUE,
#    isotype.control.name.vec = rownames(cells_mtx_rawprot)[29:31]
#    )
#  # note: normalization takes ~ 20 seconds
#  # system.time()
#  # user  system elapsed
#  #  20.799   0.209  21.783

## ---- eval = FALSE------------------------------------------------------------
#  dsb_norm_prot = apply(dsb_norm_prot, 2, function(x){ ifelse(test = x < -10, yes = 0, no = x)})

## ---- eval = FALSE------------------------------------------------------------
#  # filter raw protein, RNA and metadata to only include cell-containing droplets
#  cells_rna = rna[ ,positive_cells]
#  md2 = md[positive_cells, ]
#  
#  # create Seurat object !note: min.cells is a gene filter, not a cell filter
#  s = Seurat::CreateSeuratObject(counts = cells_rna, meta.data = md2,
#                                 assay = "RNA", min.cells = 20)
#  
#  # add dsb normalized matrix "dsb_norm_prot" to the "CITE" assay data slot
#  s[["CITE"]] = Seurat::CreateAssayObject(data = dsb_norm_prot)

## ---- eval = FALSE------------------------------------------------------------
#  # cluster and run umap (based directly on dsb normalized values without istype controls )
#  prots = rownames(s@assays$CITE@data)[1:28]
#  
#  s = FindNeighbors(object = s, dims = NULL, assay = 'CITE',
#                    features = prots, k.param = 30, verbose = FALSE)
#  
#  # direct graph clustering
#  s = FindClusters(object = s, resolution = 1, algorithm = 3, graph.name = 'CITE_snn', verbose = FALSE)
#  
#  # umap (optional)
#  s = RunUMAP(object = s, assay = "CITE", features = prots, seed.use = 1990,
#              min.dist = 0.2, n.neighbors = 30, verbose = FALSE)

## ---- eval = FALSE------------------------------------------------------------
#  # make results dataframe
#  d = cbind(s@meta.data, as.data.frame(t(s@assays$CITE@data)), s@reductions$umap@cell.embeddings)
#  
#  
#  # calculate the median protein expression separately for each cluster
#  adt_plot = d %>%
#    dplyr::group_by(CITE_snn_res.1) %>%
#    dplyr::summarize_at(.vars = prots, .funs = median) %>%
#    tibble::remove_rownames() %>%
#    tibble::column_to_rownames("CITE_snn_res.1")
#  # plot a heatmap of the average dsb normalized values for each cluster
#  pheatmap::pheatmap(t(adt_plot),
#                     color = viridis::viridis(25, option = "B"),
#                     fontsize_row = 8, border_color = NA)

## ---- eval = FALSE------------------------------------------------------------
#  
#  clusters = c(0:13)
#  celltype = c("CD4_Tcell_Memory", # 0
#                "CD14_Monocytes", #1
#                "CD14_Monocytes_Activated", #2
#                "CD4_Naive_Tcell", #3
#                "B_Cells", #4
#                "NK_Cells", #5
#                "CD4_Naive_Tcell_CD62L+", #6
#                "CD8_Memory_Tcell", #7
#                "DC", #8
#                "CD8_Naive_Tcell", #9
#                "CD4_Effector", #10
#                "CD16_Monocyte", #11
#                "DOUBLETS", #12
#                "DoubleNegative_Tcell" #13
#                )
#  s@meta.data$celltype = plyr::mapvalues(x = s@meta.data$CITE_snn_res.1, from = clusters, to = celltype)
#  
#  # plot
#  Seurat::DimPlot(s, reduction = 'umap', group.by = 'celltype',
#                label = TRUE, repel = TRUE, label.size = 2.5, pt.size = 0.1) +
#    theme_bw() + NoLegend() + ggtitle('dsb normalized protein')

## ---- eval = FALSE------------------------------------------------------------
#  # use pearson residuals as normalized values for pca
#  DefaultAssay(s) = "RNA"
#  s = NormalizeData(s, verbose = FALSE) %>%
#    FindVariableFeatures(selection.method = 'vst', verbose = FALSE) %>%
#    ScaleData(verbose = FALSE) %>%
#    RunPCA(verbose = FALSE)
#  
#  # set up dsb values to use in WNN analysis
#  DefaultAssay(s) = "CITE"
#  # hack seurat to use normalized protein values as a dimensionality reduction object.
#  VariableFeatures(s) = prots
#  
#  # run true pca to initialize dr pca slot for WNN
#  s = ScaleData(s, assay = 'CITE', verbose = FALSE)
#  s = RunPCA(s, reduction.name = 'pdsb', features = VariableFeatures(s), verbose = FALSE)
#  
#  # make matrix of norm values to add as dr embeddings
#  pseudo = t(s@assays$CITE@data)[,1:29]
#  pseudo_colnames = paste('pseudo', 1:29, sep = "_")
#  colnames(pseudo) = pseudo_colnames
#  # add to object
#  s@reductions$pdsb@cell.embeddings = pseudo
#  
#  # run WNN
#  s = FindMultiModalNeighbors(
#    object = s,
#    reduction.list = list("pca", "pdsb"),
#    weighted.nn.name = "dsb_wnn",
#    knn.graph.name = "dsb_knn",
#    modality.weight.name = "dsb_weight",
#    snn.graph.name = "dsb_snn",
#    dims.list = list(1:30, 1:29),
#    verbose = FALSE
#  )
#  
#  s = FindClusters(s, graph.name = "dsb_knn", algorithm = 3, resolution = 1.5,
#                   random.seed = 1990,  verbose = FALSE)
#  s = RunUMAP(s, nn.name = "dsb_wnn", reduction.name = "dsb_wnn_umap",
#              reduction.key = "dsb_wnnUMAP_", seed.use = 1990, verbose = FALSE)
#  
#  # plot
#  p1 = Seurat::DimPlot(s, reduction = 'dsb_wnn_umap', group.by = 'dsb_knn_res.1.5',
#                label = TRUE, repel = TRUE, label.size = 2.5, pt.size = 0.1) +
#    theme_bw() +
#    xlab('dsb protein RNA multimodal UMAP 1') +
#    ylab('dsb protein RNA multimodal UMAP 2') +
#    ggtitle('WNN using dsb normalized protein values')
#  
#  p1
#  

## ---- eval = FALSE------------------------------------------------------------
#  # create multimodal heatmap
#  vf = VariableFeatures(s,assay = "RNA")
#  
#  Idents(s) = "dsb_knn_res.1.5"
#  DefaultAssay(s)  = "RNA"
#  rnade = FindAllMarkers(s, features = vf, only.pos = TRUE)
#  gene_plot = rnade %>% filter(avg_log2FC > 1 ) %>%  group_by(cluster) %>% top_n(3) %$% gene %>% unique
#  
#  s@meta.data$celltype_subcluster = paste(s@meta.data$celltype, s@meta.data$dsb_knn_res.1.5)
#  
#  d = cbind(s@meta.data,
#            # protein
#            as.data.frame(t(s@assays$CITE@data)),
#            # mRNA
#            as.data.frame(t(as.matrix(s@assays$RNA@data[gene_plot, ]))),
#            s@reductions$umap@cell.embeddings)
#  
#  # combined data
#  adt_plot = d %>%
#    dplyr::group_by(dsb_knn_res.1.5) %>%
#    dplyr::summarize_at(.vars = c(prots, gene_plot), .funs = median) %>%
#    tibble::remove_rownames() %>%
#    tibble::column_to_rownames("dsb_knn_res.1.5")
#  
#  
#  # make a combined plot
#  suppressMessages(library(ComplexHeatmap))
#  # protein heatmap
#  prot_col = circlize::colorRamp2(breaks = seq(-10,30, by = 2), colors = viridis::viridis(n = 18, option = "B", end = 0.95))
#  p1 = Heatmap(t(adt_plot)[prots, ], name = "protein",col = prot_col, use_raster = T,
#               row_names_gp = gpar(color = "black", fontsize = 5))
#  
#  # mRNA heatmap
#  mrna = t(adt_plot)[gene_plot, ]
#  rna_col = circlize::colorRamp2(breaks = c(-2,-1,0,1,2), colors = colorspace::diverge_hsv(n = 5))
#  p2 = Heatmap(t(scale(t(mrna))), name = "mRNA", col = rna_col, use_raster = T,
#               clustering_method_columns = 'average',
#               column_names_gp = gpar(color = "black", fontsize = 7),
#               row_names_gp = gpar(color = "black", fontsize = 5))
#  
#  ht_list = p1 %v% p2
#  draw(ht_list)
#  

## ---- eval=FALSE--------------------------------------------------------------
#  # use pearson residuals as normalized values for pca
#  DefaultAssay(s) = "RNA"
#  s = NormalizeData(s, verbose = FALSE) %>%
#    FindVariableFeatures(selection.method = 'vst', verbose = FALSE) %>%
#    ScaleData(verbose = FALSE) %>%
#    RunPCA(verbose = FALSE)
#  
#  # set up dsb values to use in WNN analysis (do not normalize with CLR, use dsb normalized values)
#  DefaultAssay(s) = "CITE"
#  VariableFeatures(s) = prots
#  s = s %>% ScaleData() %>% RunPCA(reduction.name = 'apca')
#  
#  # run WNN
#  s = FindMultiModalNeighbors(
#    s, reduction.list = list("pca", "apca"),
#    dims.list = list(1:30, 1:18),
#    modality.weight.name = "RNA.weight"
#  )
#  
#  # cluster
#  s <- RunUMAP(s, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
#  s <- FindClusters(s, graph.name = "wsnn", algorithm = 3, resolution = 1.5, verbose = FALSE, random.seed = 1990)
#  
#  p1 = Seurat::DimPlot(s, reduction = 'wnn.umap', group.by = 'wsnn_res.1.5',
#                label = TRUE, repel = TRUE, label.size = 2.5, pt.size = 0.1) +
#    theme_bw() +
#    xlab('Default WNN with dsb values UMAP 1') +
#    ylab('Default WNN with dsb values UMAP 2')
#  
#  p1
#  

## ---- eval=FALSE--------------------------------------------------------------
#  suppressMessages(library(SingleCellExperiment))
#  sce = SingleCellExperiment(assays = list(counts = cells_rna), colData = md2)
#  # define the dsb normalized values as logcounts to use a common SingleCellExperiment / Bioconductor convention
#  adt = SummarizedExperiment(assays = list('counts' = as.matrix(cells_mtx_rawprot), 'logcounts' = as.matrix(dsb_norm_prot)))
#  altExp(sce, "CITE") = adt

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
#  # proceed with same tutorial workflow shown above.

## ---- eval = FALSE------------------------------------------------------------
#  # find outliers
#  pheatmap::pheatmap(apply(dsb_norm_prot, 1, function(x){
#    quantile(x,c(0.9999, 0.99, 0.98, 0.95, 0.0001, 0.01, 0.1))
#    }))
#  

## ---- eval = FALSE------------------------------------------------------------
#  dsb_norm_prot = DSBNormalizeProtein(
#                             cell_protein_matrix = cells_citeseq_mtx,
#                             empty_drop_matrix = empty_drop_citeseq_mtx,
#                             denoise.counts = TRUE,
#                             use.isotype.control = TRUE,
#                             isotype.control.name.vec = rownames(cells_citeseq_mtx)[67:70],
#                             # implement Quantile clipping
#                             quantile.clipping = TRUE,
#                             # high and low otlier quantile across proteins to clip
#                             quantile.clip = c(0.001, 0.9995)
#                             )

## ---- eval = FALSE------------------------------------------------------------
#  dsb_norm_prot = apply(dsb_norm_prot, 2, function(x){
#    ifelse(test = x < -10, yes = 0, no = x)
#    })

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

