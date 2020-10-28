library(readr)
library(tidyr)
library(dplyr)
library(ggplot2)
library(forcats)

load_image_data <- function(file_name) {
  d <- read_tsv(file_name, col_types=cols(.default='d'), col_names=F) %>% as.matrix()
  colnames(d) <- 1:ncol(d)
  d
}

load_expr_data <- function(file_name, genes_file_name) {
  genes <- read_lines(genes_file_name)
  read_tsv(file_name, col_types=cols(.default='d'), col_names=genes) %>% as.matrix()
}

SCCA <- function(datasets, penalty_x, penalty_z, K) {
  datasets[[1]] <- scale(datasets[[1]])
  datasets[[2]] <- scale(datasets[[2]])
  x <- PMA::CCA(datasets[[1]], datasets[[2]], penaltyx=penalty_x, penaltyz=penalty_z, K=K, standardize=F)
  rownames(x$u) <- colnames(datasets[[1]])
  rownames(x$v) <- colnames(datasets[[2]])
  x$group_names <- names(datasets)
  x$CCA_var1 <- datasets[[1]] %*% x$u
  x$CCA_var2 <- datasets[[2]] %*% x$v
  x
}

SCCA_coefs <- function(cca) {
  data_frame(CCA_var = 1:cca$K) %>%
    group_by(CCA_var) %>%
    do({
      k <- .$CCA_var
      nonzero_1 <- which(cca$u[, k] != 0)
      nonzero_2 <- which(cca$v[, k] != 0)
      bind_rows(
        data_frame(
          type = cca$group_names[1],
          name = rownames(cca$u)[nonzero_1],
          coefficient = cca$u[nonzero_1, k]
        ),
        
        data_frame(
          type = cca$group_names[2],
          name = rownames(cca$v)[nonzero_2],
          coefficient = cca$v[nonzero_2, k]
        )
      )
    }) %>%
    ungroup
}

SCCA_cors <- function(cca) data_frame(k = 1:length(cca$d), d = cca$d)

args <- commandArgs(trailingOnly=TRUE)
indir <- args[1]
outdir <- args[2]
K <- as.integer(args[3])

dir.create(outdir, showWarnings = FALSE)

### CAE3 PCA

image <- load_image_data(paste(indir,"/CCA_input_image.txt", sep =""))
expr <- load_expr_data(paste(indir,"/CCA_input_expr.txt", sep =""), paste(indir,"/CCA_genes.txt", sep =""))

cca <- SCCA(list(gene = expr, image = image), penalty_x=0.05, penalty_z=0.15, K=K)
cca$CCA_var1 %>% as_data_frame() %>% write_tsv(paste(outdir,"/CCA_ls_canonvar_expr.txt", sep =""), col_names=F)
cca$CCA_var2 %>% as_data_frame() %>% write_tsv(paste(outdir,"/CCA_ls_canonvar_image.txt", sep =""), col_names=F)
cca %>% SCCA_coefs() %>% write_tsv(paste(outdir,"/CCA_ls_coefs.txt", sep =""))


### CAE3 discriminative PCA

image <- load_image_data(paste(indir,"/discrim_CCA_input_image.txt", sep =""))
expr <- load_expr_data(paste(indir,"/discrim_CCA_input_expr.txt", sep =""), paste(indir,"/discrim_CCA_genes.txt", sep =""))

cca <- SCCA(list(gene = expr, image = image), penalty_x=0.05, penalty_z=0.15, K=ncol(image))
cca$CCA_var1 %>% as_data_frame() %>% write_tsv(paste(outdir,"/discrim_CCA_ls_canonvar_expr.txt", sep =""), col_names=F)
cca$CCA_var2 %>% as_data_frame() %>% write_tsv(paste(outdir,"/discrim_CCA_ls_canonvar_image.txt", sep =""), col_names=F)
cca %>% SCCA_coefs() %>% write_tsv(paste(outdir,"/discrim_CCA_ls_coefs.txt", sep =""))


##### shuffle data and see how SCCA correlation is affected

data_mods <- list('Original' = function(m) m,
                  'Shuffle Samples' = function(m) m[sample(nrow(m)), ],
                  'Shuffle Within Genes' = function(m) apply(m, 2, sample),
                  'Normal Random' = function(m) matrix(rnorm(nrow(m) * ncol(m)), nrow=nrow(m), ncol=ncol(m)))

cca_mod <- crossing(mod = factor(names(data_mods), levels=names(data_mods)),
                    iter = 1:10) %>%
  group_by(mod, iter) %>%
  do({
    cca <- SCCA(list(gene = data_mods[[.$mod]](expr), image = image), penalty_x=0.05, penalty_z=0.15, K=1)
    SCCA_cors(cca) %>%
      rowwise() %>%
      mutate(cor = cor(cca$CCA_var1[, k], cca$CCA_var2[, k])) %>%
      ungroup()
  }) %>%
  ungroup

write_tsv(cca_mod, paste(outdir, "/CCA_expr_rand.txt", sep =""))

cca_mod <- read_tsv(paste(outdir,"/CCA_expr_rand.txt", sep =""), col_types='ciidd') %>%
  mutate(mod = mod %>%
           fct_recode('Original' = 'original',
                      'Shuffle samples' = 'shuffle samples',
                      'Shuffle within genes' = 'shuffle within genes',
                      'Normal random' = 'normal random') %>%
           fct_relevel('Original',
                       'Shuffle samples',
                       'Shuffle within genes',
                       'Normal random'))

cca_mod %>%
  mutate(mod = fct_rev(mod)) %>%
  ggplot(aes(x=mod, y=d)) +
  geom_boxplot() +
  xlab("Input randomization") +
  ylab("Dot product of first pair of CCA variables") +
  ylim(0, NA) +
  coord_flip()
ggsave(paste(outdir,"/CCA_expr_rand.pdf", sep =""), width=6, height=2)

cca_mod %>%
  group_by(mod) %>%
  summarise(d_mean = mean(d),
            d_stdev = sd(d),
            cor_mean = mean(cor),
            cor_stdev = sd(cor))
