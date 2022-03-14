## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = FALSE,
  # fig.path = "man/figures/README-",
  out.width = "100%"
)

## ---- eval = FALSE------------------------------------------------------------
#  install.packages('dsb')
#  library(dsb)
#  
#  adt_norm = DSBNormalizeProtein(
#    cell_protein_matrix = cells_citeseq_mtx,
#    empty_drop_matrix = empty_drop_citeseq_mtx,
#    denoise.counts = TRUE,
#    use.isotype.control = TRUE,
#    isotype.control.name.vec = rownames(cells_citeseq_mtx)[67:70]
#    )

## ---- eval = FALSE------------------------------------------------------------
#  library(dsb)
#  
#  # read raw data using the Seurat function "Read10X"
#  raw = Seurat::Read10X("data/raw_feature_bc_matrix/")
#  cells = Seurat::Read10X("data/filtered_feature_bc_matrix/")
#  
#  # define cell-containing barcodes and separate cells and empty drops
#  stained_cells = colnames(cells$`Gene Expression`)
#  background = setdiff(colnames(raw$`Gene Expression`), stained_cells)
#  
#  # split the data into separate matrices for RNA and ADT
#  prot = raw$`Antibody Capture`
#  rna = raw$`Gene Expression`

## ---- eval = FALSE------------------------------------------------------------
#  # create metadata of droplet QC stats used in standard scRNAseq processing
#  mtgene = grep(pattern = "^MT-", rownames(rna), value = TRUE) # used below
#  
#  md = data.frame(
#    rna.size = log10(Matrix::colSums(rna)),
#    prot.size = log10(Matrix::colSums(prot)),
#    n.gene = Matrix::colSums(rna > 0),
#    mt.prop = Matrix::colSums(rna[mtgene, ]) / Matrix::colSums(rna)
#  )
#  # add indicator for barcodes Cell Ranger called as cells
#  md$drop.class = ifelse(rownames(md) %in% stained_cells, 'cell', 'background')
#  
#  # remove barcodes with no evidence of capture in the experiment
#  md = md[md$rna.size > 0 & md$prot.size > 0, ]
#  

## ---- eval = FALSE------------------------------------------------------------
#  background_drops = rownames(
#    md[ md$prot.size > 1.5 &
#        md$prot.size < 3 &
#        md$rna.size < 2.5, ]
#    )
#  background.adt.mtx = as.matrix(prot[ , background_drops])

## ---- eval = FALSE------------------------------------------------------------
#  
#  # calculate statistical thresholds for droplet filtering.
#  cellmd = md[md$drop.class == 'cell', ]
#  
#  # filter drops with + / - 3 median absolute deviations from the median library size
#  rna.mult = (3*mad(cellmd$rna.size))
#  prot.mult = (3*mad(cellmd$prot.size))
#  rna.lower = median(cellmd$rna.size) - rna.mult
#  rna.upper = median(cellmd$rna.size) + rna.mult
#  prot.lower = median(cellmd$prot.size) - prot.mult
#  prot.upper = median(cellmd$prot.size) + prot.mult
#  
#  # filter rows based on droplet qualty control metrics
#  qc_cells = rownames(
#    cellmd[cellmd$prot.size > prot.lower &
#           cellmd$prot.size < prot.upper &
#           cellmd$rna.size > rna.lower &
#           cellmd$rna.size < rna.upper &
#           cellmd$mt.prop < 0.14, ]
#    )

## ---- eval = FALSE------------------------------------------------------------
#  length(qc_cells)

## ---- eval = FALSE------------------------------------------------------------
#  cell.adt.raw = as.matrix(prot[ , qc_cells])
#  cell.rna.raw = rna[ ,qc_cells]
#  cellmd = cellmd[qc_cells, ]

## ---- eval = FALSE------------------------------------------------------------
#  # flter
#  pm = sort(apply(cell.adt.raw, 1, max))
#  pm2 = apply(background.adt.mtx, 1, max)
#  head(pm)

## ---- eval = FALSE------------------------------------------------------------
#  # remove the non staining protein
#  cell.adt.raw  = cell.adt.raw[!rownames(cell.adt.raw) == 'CD34_TotalSeqB', ]
#  background.adt.mtx = background.adt.mtx[!rownames(background.adt.mtx) == 'CD34_TotalSeqB', ]
#  

## ---- eval = FALSE------------------------------------------------------------
#  # define isotype controls
#  isotype.controls = c("IgG1_control_TotalSeqB", "IgG2a_control_TotalSeqB",
#                       "IgG2b_control_TotalSeqB")
#  
#  # normalize and denoise with dsb with
#  cells.dsb.norm = DSBNormalizeProtein(
#    cell_protein_matrix = cell.adt.raw,
#    empty_drop_matrix = background.adt.mtx,
#    denoise.counts = TRUE,
#    use.isotype.control = TRUE,
#    isotype.control.name.vec = isotype.controls
#    )
#  # note: normalization takes ~ 20 seconds
#  # system.time()
#  # user  system elapsed
#  #  20.799   0.209  21.783

## ---- eval = FALSE------------------------------------------------------------
#  # dsb with non-standard options
#  cell.adt.dsb.2 = DSBNormalizeProtein(
#    cell_protein_matrix = cell.adt.raw,
#    empty_drop_matrix = background.adt.mtx,
#    denoise.counts = TRUE,
#    use.isotype.control = TRUE,
#    isotype.control.name.vec = rownames(cell.adt.raw)[29:31],
#    define.pseudocount = TRUE,
#    pseudocount.use = 1,
#    scale.factor = 'mean_subtract',
#    quantile.clipping = TRUE,
#    quantile.clip = c(0.01, 0.99),
#    return.stats = TRUE
#    )
#  
#  

## ---- eval = FALSE------------------------------------------------------------
#  # Seurat workflow
#  library(Seurat)
#  
#  # integrating with Seurat
#  stopifnot(isTRUE(all.equal(rownames(cellmd), colnames(cell.adt.raw))))
#  stopifnot(isTRUE(all.equal(rownames(cellmd), colnames(cell.rna.raw))))
#  
#  # create Seurat object note: min.cells is a gene filter, not a cell filter
#  s = Seurat::CreateSeuratObject(counts = cell.rna.raw,
#                                 meta.data = cellmd,
#                                 assay = "RNA",
#                                 min.cells = 20)
#  
#  # add dsb normalized matrix "cell.adt.dsb" to the "CITE" data (not counts!) slot
#  s[["CITE"]] = Seurat::CreateAssayObject(data = cells.dsb.norm)
#  

## ---- eval = FALSE------------------------------------------------------------
#  # define proteins to use in clustering (non-isptype controls)
#  prots = rownames(s@assays$CITE@data)[1:28]
#  
#  # cluster and run umap
#  s = Seurat::FindNeighbors(object = s, dims = NULL,assay = 'CITE',
#                            features = prots, k.param = 30,
#                            verbose = FALSE)
#  
#  # direct graph clustering
#  s = Seurat::FindClusters(object = s, resolution = 1,
#                           algorithm = 3,
#                           graph.name = 'CITE_snn',
#                           verbose = FALSE)
#  # umap (optional)
#  # s = Seurat::RunUMAP(object = s, assay = "CITE", features = prots,
#  #                     seed.use = 1990, min.dist = 0.2, n.neighbors = 30,
#  #                     verbose = FALSE)
#  
#  # make results dataframe
#  d = cbind(s@meta.data,
#            as.data.frame(t(s@assays$CITE@data))
#            # s@reductions$umap@cell.embeddings)
#            )

## ---- eval = FALSE------------------------------------------------------------
#  library(magrittr)
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
#  

## ---- eval = FALSE------------------------------------------------------------
#  clusters = c(0:13)
#  celltype = c("CD4_Tcell_Memory", # 0
#               "CD14_Monocytes", #1
#               "CD14_Monocytes_Activated", #2
#               "CD4_Naive_Tcell", #3
#               "B_Cells", #4
#               "NK_Cells", #5
#               "CD4_Naive_Tcell_CD62L+", #6
#               "CD8_Memory_Tcell", #7
#               "DC", #8
#               "CD8_Naive_Tcell", #9
#               "CD4_Effector", #10
#               "CD16_Monocyte", #11
#               "DOUBLETS", #12
#               "DoubleNegative_Tcell" #13
#  )
#  
#  s@meta.data$celltype = plyr::mapvalues(
#    x = s@meta.data$CITE_snn_res.1,
#    from = clusters,  to = celltype
#    )
#  
#  # # optional -- dimensionality reduction plot
#  # Seurat::DimPlot(s, reduction = 'umap', group.by = 'celltype',
#  #               label = TRUE, repel = TRUE, label.size = 2.5, pt.size = 0.1) +
#  #   theme_bw() + NoLegend() + ggtitle('dsb normalized protein')

## -----------------------------------------------------------------------------
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
#  s = s %>%
#    ScaleData() %>%
#    RunPCA(reduction.name = 'apca', verbose = FALSE)
#  
#  # run WNN
#  s = FindMultiModalNeighbors(
#    s, reduction.list = list("pca", "apca"),
#    dims.list = list(1:30, 1:18),
#    modality.weight.name = "RNA.weight",
#    verbose = FALSE
#  )
#  
#  # cluster
#  s <- FindClusters(s, graph.name = "wsnn",
#                    algorithm = 3,
#                    resolution = 1.5,
#                    verbose = FALSE,
#                    random.seed = 1990)

## -----------------------------------------------------------------------------
#  ## WNN with dsb values
#  # use RNA pearson residuals as normalized values for RNA pca
#  DefaultAssay(s) = "RNA"
#  s = NormalizeData(s, verbose = FALSE) %>%
#    FindVariableFeatures(selection.method = 'vst', verbose = FALSE) %>%
#    ScaleData(verbose = FALSE) %>%
#    RunPCA(verbose = FALSE)
#  
#  
#  # set up dsb values to use in WNN analysis
#  DefaultAssay(s) = "CITE"
#  VariableFeatures(s) = prots
#  
#  # run true pca to initialize dr pca slot for WNN
#  ## Not used {
#  s = ScaleData(s, assay = 'CITE', verbose = FALSE)
#  s = RunPCA(s, reduction.name = 'pdsb',features = VariableFeatures(s), verbose = FALSE)
#  # }
#  
#  # make matrix of normalized protein values to add as dr embeddings
#  pseudo = t(GetAssayData(s,slot = 'data',assay = 'CITE'))[,1:29]
#  colnames(pseudo) = paste('pseudo', 1:29, sep = "_")
#  s@reductions$pdsb@cell.embeddings = pseudo
#  
#  # run WNN directly using dsb normalized values.
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
#  # cluster based on the join RNA and protein graph
#  s = FindClusters(s, graph.name = "dsb_knn",
#                   algorithm = 3,
#                   resolution = 1.5,
#                   random.seed = 1990,
#                   verbose = FALSE)

## -----------------------------------------------------------------------------
#  # create multimodal heatmap
#  vf = VariableFeatures(s,assay = "RNA")
#  
#  # find marker genes for the joint clusters
#  Idents(s) = "dsb_knn_res.1.5"
#  DefaultAssay(s)  = "RNA"
#  rnade = FindAllMarkers(s, features = vf, only.pos = TRUE, verbose = FALSE)
#  gene_plot = rnade %>%
#    dplyr::filter(avg_log2FC > 1 ) %>%
#    dplyr::group_by(cluster) %>%
#    dplyr::top_n(3) %$% gene %>% unique
#  
#  
#  cite_data = GetAssayData(s,slot = 'data',assay = 'CITE') %>% t()
#  rna_subset = GetAssayData(s,assay = 'RNA',slot = 'data')[gene_plot, ] %>%
#    as.data.frame() %>%
#    t() %>%
#    as.matrix()
#  
#  # combine into dataframe
#  d = cbind(s@meta.data, cite_data, rna_subset)
#  
#  # calculate the median protein expression per cluster
#  dat_plot = d %>%
#    dplyr::group_by(dsb_knn_res.1.5) %>%
#    dplyr::summarize_at(.vars = c(prots, gene_plot), .funs = median) %>%
#    tibble::remove_rownames() %>%
#    tibble::column_to_rownames("dsb_knn_res.1.5")
#  

## ---- fig.height=4, fig.width=3-----------------------------------------------
#  # make a combined plot
#  suppressMessages(library(ComplexHeatmap)); ht_opt$message = FALSE
#  
#  # protein heatmap
#  # protein heatmap
#  prot_col = circlize::colorRamp2(breaks = seq(-1,25, by = 1),
#                                  colors = viridis::viridis(n = 27, option = "B"))
#  p1 = Heatmap(t(dat_plot)[prots, ],
#               name = "protein",
#               col = prot_col,
#               use_raster = T,
#               row_names_gp = gpar(color = "black", fontsize = 5)
#  )
#  
#  
#  # mRNA heatmap
#  mrna = t(dat_plot)[gene_plot, ]
#  rna_col = circlize::colorRamp2(breaks = c(-2,-1,0,1,2),
#                                 colors = colorspace::diverge_hsv(n = 5))
#  p2 = Heatmap(t(scale(t(mrna))),
#               name = "mRNA",
#               col = rna_col,
#               use_raster = T,
#               clustering_method_columns = 'average',
#               column_names_gp = gpar(color = "black", fontsize = 7),
#               row_names_gp = gpar(color = "black", fontsize = 5))
#  
#  
#  # combine heatmaps
#  ht_list = p1 %v% p2
#  draw(ht_list)
#  

## ---- eval = FALSE------------------------------------------------------------
#  library(Seurat)
#  umi = Read10X(data.dir = 'data/raw_feature_bc_matrix/')
#  k = 3
#  barcode.whitelist =
#    rownames(
#      CreateSeuratObject(counts = umi,
#                         min.features = k,  # retain all barcodes with at least k raw mRNA
#                         min.cells = 800, # this just speeds up the function by removing genes.
#                         )@meta.data
#      )
#  
#  write.table(barcode.whitelist,
#  file =paste0(your_save_path,"barcode.whitelist.tsv"),
#  sep = '\t', quote = FALSE, col.names = FALSE, row.names = FALSE)
#  

