library(tidyverse)
library(stringr)

prepare_CCA_files <- function(expr, images, samples, expr_outfile, image_outfile, gene_outfile, image_name_outfile) {
  ### only use samples with expression and image data ###
  samples <- samples %>%
    filter(sample %in% rownames(expr),
           image %in% rownames(images))
    
    
  images <- images[samples$image, ]
  expr <- expr[samples$sample, ]

  # required for CCA:
  images <- images[, sapply(1:ncol(images), function(x) sd(images[, x]) > 0)]
  expr <- expr[, sapply(1:ncol(expr), function(x) sd(expr[, x]) > 0)]
                        
  
  # log and scale expression data, scale image data
  expr <- log(expr + 1) %>% scale()
  images <- scale(images)
                        
  
  ### save gene and image file order for analyzing CCA output ###
  write_lines(colnames(expr), gene_outfile)
  write_lines(rownames(images), image_name_outfile)
  
  ### save image and expression matrices ###
  expr %>% as_data_frame %>% write_tsv(expr_outfile, col_names=F)
  images %>% as_data_frame %>% write_tsv(image_outfile, col_names=F)
  images
}

PCA_whiten <- function(x) {
  pca <- prcomp(x)$x
  sapply(1:ncol(pca), function(c) pca[, c] / sd(pca[, c]))
}
         
args <- commandArgs(trailingOnly=TRUE)
sample_file <- args[1]
expr_file <- args[2]
image_file <- args[3]
outdir <- args[4]
         
dir.create(outdir, showWarnings = FALSE)
         

samples <- read_tsv(sample_file, col_types='cc')

expr <- read.csv(expr_file,row.names = 1,header = TRUE) %>% as.matrix()

images <- read.csv(image_file, header=F, row.names=1) %>% as.matrix()
images_CAE3_discrim <- prepare_CCA_files(
  expr = expr,
  images = images,
  samples = samples,
  expr_outfile = paste(outdir,"/CCA_input_expr.txt", sep =""),
  image_outfile = paste(outdir,"/CCA_input_image.txt", sep =""),
  gene_outfile = paste(outdir,"/CCA_genes.txt", sep =""),
  image_name_outfile = paste(outdir,"/CCA_images.txt", sep ="")
)

images_discrim <- PCA_whiten(images_CAE3_discrim)
x <- prepare_CCA_files(
  expr = expr,
  images = images_discrim,
  samples = samples,
  expr_outfile = paste(outdir,"/discrim_CCA_input_expr.txt", sep =""),
  image_outfile = paste(outdir,"/discrim_CCA_input_image.txt", sep =""),
  gene_outfile = paste(outdir,"/discrim_CCA_genes.txt", sep =""),
  image_name_outfile = paste(outdir,"/discrim_CCA_images.txt", sep ="")
)