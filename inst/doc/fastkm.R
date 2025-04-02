## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----eval = FALSE-------------------------------------------------------------
#  library(dsb)
#  library(MASS)
#  library(mclust)

## ----eval = FALSE-------------------------------------------------------------
#  
#  isotypes = c("MouseIgG1kappaisotype_PROT", "MouseIgG2akappaisotype_PROT",
#               "Mouse IgG2bkIsotype_PROT", "RatIgG2bkIsotype_PROT")
#  
#  norm.adt = ModelNegativeADTnorm(
#    cell_protein_matrix = cells_citeseq_mtx,
#    fast.km = TRUE,
#    denoise.counts = TRUE,
#    use.isotype.control = TRUE,
#    isotype.control.name.vec = isotypes
#    )
#  

## ----eval = FALSE-------------------------------------------------------------
#  
#  norm.adt = DSBNormalizeProtein(
#    cell_protein_matrix = dsb::raw.adt.matrix,
#    empty_drop_matrix = dsb::empty_drop_citeseq_mtx,
#    fast.km = TRUE,
#    denoise.counts = TRUE,
#    use.isotype.control = TRUE,
#    isotype.control.name.vec = isotypes
#    )
#  

## ----eval = FALSE-------------------------------------------------------------
#  r = "deepskyblue3"
#  library(dsb)
#  
#  # specify isotypes
#  isotypes.names = rownames(cells_citeseq_mtx)[67:70]
#  
#  norm = DSBNormalizeProtein(
#    # set fast.km = TRUE to run the fast method
#    fast.km = TRUE,
#    cell_protein_matrix = dsb::cells_citeseq_mtx,
#    empty_drop_matrix = dsb::empty_drop_citeseq_mtx,
#    denoise.counts = TRUE,
#    use.isotype.control = TRUE,
#    isotype.control.name.vec = rownames(cells_citeseq_mtx)[67:70],
#  )
#  
#  # original method
#  norm.original = dsb::DSBNormalizeProtein(
#    cell_protein_matrix = dsb::cells_citeseq_mtx,
#    empty_drop_matrix = dsb::empty_drop_citeseq_mtx,
#    denoise.counts = TRUE,
#    use.isotype.control = TRUE,
#    isotype.control.name.vec = rownames(cells_citeseq_mtx)[67:70],
#  )
#  
#  n.original = norm.original$dsb_normalized_matrix
#  n.fast = norm$dsb_normalized_matrix
#  # individual correlations
#  par(mfrow=c(1,2))
#  plot(n.original['CD8_PROT', ], n.fast['CD8_PROT', ],
#       pch = 16,
#       font.main = 1,
#       col = adjustcolor(r, alpha.f = 0.2),
#       cex = 0.6,
#       xlab = "dsb original",
#       ylab = "dsb km.fast",
#       main = 'CD8 Normalized ADT'
#  )
#  plot(n.original['CD4_PROT', ], n.fast['CD4_PROT', ],
#       pch = 16, font.main = 1, cex = 0.6,
#       col = adjustcolor(r, alpha.f = 0.2),
#       xlab = "dsb original",
#       ylab = "dsb km.fast",
#       main = 'CD4 Normalized ADT'
#  )
#  

## ----eval = FALSE-------------------------------------------------------------
#  
#  correlations <- sapply(seq_len(nrow(n.original)), function(x){
#    cor(n.original[x, ], n.fast[x, ], method = 'pearson')
#    })
#  
#  # plot
#  hist(correlations, breaks = 20, xlim = c(0.97, 1),
#       main = "correlation per protein\n km.fast vs original method",
#       font.main = 1,
#       xlab = "Pearson correlation", freq = FALSE, col = "lightgray", border = "white")
#  rug(correlations)
#  

