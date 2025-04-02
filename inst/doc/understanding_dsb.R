## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## -----------------------------------------------------------------------------
suppressMessages(library(mclust))
suppressMessages(library(magrittr))
suppressMessages(library(ggplot2))
library(dsb) 

## -----------------------------------------------------------------------------
cell = dsb::cells_citeseq_mtx
neg = dsb::empty_drop_citeseq_mtx

## -----------------------------------------------------------------------------
# log transformation
dlog = log(cell + 10)
nlog = log(neg + 10)

## -----------------------------------------------------------------------------
# calc mean and sd of background drops 
sd_nlog = apply(nlog, 1 , sd)
mean_nlog = apply(nlog, 1 , mean)


## -----------------------------------------------------------------------------

norm_adt = apply(dlog, 2, function(x) (x  - mean_nlog) / sd_nlog) 


## ----fig.width=8, fig.height=3.5----------------------------------------------
# check structure of denoised data with zero centering of background population 
r = 'deepskyblue3'
plist = list(theme_bw(), geom_density_2d(color = r), 
             geom_vline(xintercept = 0, linetype = 'dashed'),
             geom_hline(yintercept = 0, linetype = 'dashed'),
             xlab('CD3'), ylab('CD19')
             )

p1 = qplot(as.data.frame(t(norm_adt))$CD3_PROT, 
      as.data.frame(t(norm_adt))$CD19_PROT) + 
  plist +
  ggtitle('dsb step 1')
  
# raw data 
p2 = qplot(as.data.frame(t(cell))$CD3_PROT, 
      as.data.frame(t(cell))$CD19_PROT) + 
  plist + 
  ggtitle('RAW data') 
  

# log transformed data 
p3 = qplot(as.data.frame(t(dlog))$CD3_PROT, 
          as.data.frame(t(dlog))$CD19_PROT) + 
  plist + 
  ggtitle('log transformed') 
  
# examine distributions. 
cowplot::plot_grid(p1, p2, p3, nrow = 1) 

## -----------------------------------------------------------------------------
# fit 2 component Gaussian mixture to each cell 
cmd = apply(norm_adt, 2, function(x) {
  g = Mclust(x, G=2,  warn = TRUE , verbose = FALSE)  
  return(g) 
})


## ----fig.width=4.5, fig.height=3.5--------------------------------------------
cell.name = colnames(norm_adt)[2]
# get density 
cell2 = MclustDR(cmd[[2]])
plot(cell2, what = "density", main = 'test')
title(cex.main = 0.6, paste0('single cell:  ',cell.name,
                             '\n k = 2 component Gaussian Mixture '))


## -----------------------------------------------------------------------------
table(cell2$classification)

## ----fig.width=4.5, fig.height=3.5--------------------------------------------
# fit a mixture with between 1 and 6 components. 
m.list = lapply(as.list(1:6), function(k) Mclust(norm_adt[ ,2], G = k,
                                                 warn = FALSE, verbose = FALSE ))
# extract densities for k = 3 and k = 6 component models 
dr_3 = MclustDR(m.list[[3]])
dr_6 = MclustDR(m.list[[6]])

# visualize distributiion of protein populations with different k 
# in Gaussian mixture for a single cell 
plot.title = paste0('single cell:  ', cell.name, 
                    '\n k-component Gaussian Mixture ')

plot(dr_3,what = "density")
title(cex.main = 0.6, plot.title)


plot(dr_6,what = "density")
title(cex.main = 0.6, plot.title)


## ----fig.width=4.5, fig.height=3.5--------------------------------------------

plot(
  sapply(m.list, function(x) x$bic), type = 'b',
  ylab = 'Mclust BIC -- higher is better', 
  xlab = 'mixing components k in Gaussian Mixture model',
  main = paste0('single cell:  ', cell.name),
  col =r, pch = 18, cex = 1.4, cex.main = 0.8
  )

## ----fig.width=4.5, fig.height=3.5--------------------------------------------

# extract mu 1 as a vector 
mu.1 = unlist(
  lapply(
    cmd, function(x){x$parameters$mean[1] }
    ) 
)

# check distribution of the fitted value for µ1 across cells 
hist(mu.1, col = r,breaks = 30) 


## -----------------------------------------------------------------------------
# define isotype controls 
isotype.control.name.vec = c("Mouse IgG2bkIsotype_PROT", "MouseIgG1kappaisotype_PROT", 
                             "MouseIgG2akappaisotype_PROT", "RatIgG2bkIsotype_PROT" )
# construct noise matrix 
noise_matrix = rbind(mu.1, norm_adt[isotype.control.name.vec, ])

# transpose to fit PC 
tmat = t(noise_matrix)

# view the noise matrix on which to calculate pc1 scores. 
head(tmat)

## ----fig.width=4.5, fig.height=3.5--------------------------------------------
# calculate principal component 1 
g = prcomp(tmat, scale = TRUE)

# get the dsb technical component for each cell -- PC1 scores (position along PC1) for each cell 
head(g$x)  
head(g$rotation)
dsb.technical.component = g$x[ ,1]

hist(dsb.technical.component, breaks = 40, col = r)

## ----fig.width=4.5, fig.height=3.5--------------------------------------------

# constructs a matrix of 1 by number of columns in norm_adt1
covariate = as.matrix(dsb.technical.component)
design = matrix(1, ncol(norm_adt), 1)

# fit a linear model to solve the coefficient for each protein with QR decomposition. 
fit <- limma::lmFit(norm_adt, cbind(design, covariate))

# this subset just extracts the beta for each protein. 
beta <- fit$coefficients[, -(1:ncol(design)), drop = FALSE]
beta[is.na(beta)] <- 0

# beta is the coefficient for each prot. 
plot(beta, col = r , pch = 16, xlab = 'protein')


## -----------------------------------------------------------------------------
denoised_adt_2 = as.matrix(norm_adt) - beta %*% t(covariate)

## -----------------------------------------------------------------------------
# regress out the technical component using limma 
# note this is how limma (and dsb) calculates this 
denoised_adt = limma::removeBatchEffect(norm_adt, covariates = dsb.technical.component)


## -----------------------------------------------------------------------------
# default dsb call 
denoised_adt_3 = DSBNormalizeProtein(cell_protein_matrix = cell, 
                                     empty_drop_matrix = neg, 
                                     denoise.counts = TRUE, 
                                     isotype.control.name.vec = isotype.control.name.vec)


## ----fig.width=4.5, fig.height=3.5--------------------------------------------

qplot(as.data.frame(t(denoised_adt))$CD3_PROT, 
      as.data.frame(t(denoised_adt))$CD19_PROT) + 
  plist +
  ggtitle('dsb step 1 + 2 normalized and denoised') + 
  geom_vline(xintercept = 3.5, color = 'red') +
  geom_hline(yintercept = 3.5, color = 'red')
  

