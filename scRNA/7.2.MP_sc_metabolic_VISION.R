# =============================================================================
# 巨噬细胞代谢通路VISION分析脚本
# =============================================================================
# 功能：巨噬细胞亚群代谢通路分析，使用VISION方法计算KEGG代谢通路活性

print("Hello world!")
rm(list = ls())

# 加载配置文件
source("../config.R")

# 设置工作目录
setwd(ENV_DIR)
lf <- list.files("./")
for (file in lf) {
  source(file)
}
rm(lf, file)

# 设置项目目录
setwd(PROJECT_ROOT)

# 创建代谢分析目录
dir.create(file.path(OUTPUT_DIR, "metabolic"), showWarnings = FALSE, recursive = TRUE)
setwd(file.path(OUTPUT_DIR, "metabolic"))

library(scMetabolism)
library(ggplot2)
library(rsvd)
library(RImagePalette)
library(imager)
library(scales)
library(ggplot2)
library(ggprism)
library(reshape)
library(ggalluvial)
library(plotly)
library(paletteer)

# 读取巨噬细胞数据
MP <- qread(file.path(OUTPUT_DIR, "metabolic", "MP.qs"))

sce3 <- MP
genes <- rownames(MP@assays[["RNA"]])
genes <- toupper(genes)
genes
rownames(sce3@assays[["RNA"]]@counts) <- genes
rownames(sce3@assays[["RNA"]]@data) <- genes
countexp.Seurat <- sc.metabolism.Seurat(obj = sce3,  #Seuratde单细胞object
                                        method = "VISION", 
                                        imputation = F, 
                                        ncores = 1, 
                                        metabolism.type = "KEGG")
rm(sce3)
score <- countexp.Seurat@assays$METABOLISM$score
score[1:4,1:4]

#将score中barcode的点转为下划线
score_change <- score %>% 
  select_all(~str_replace_all(., "\\.", "-"))  #基因ID不规范会报错,下划线替换-
#确定细胞barcode椅子
identical(colnames(score_change) , rownames(countexp.Seurat@meta.data))
#[1] TRUE
countexp.Seurat@meta.data <- cbind(countexp.Seurat@meta.data,t(score_change) )

qsave(countexp.Seurat, "countexp.Seurat_MP_VISION.qs")
countexp.Seurat_MP_VISION <- qread("countexp.Seurat_MP_VISION.qs")
rm(countexp.Seurat)

#直接指定
input.pathway<-c("Oxidative phosphorylation", 
                 "Glycolysis / Gluconeogenesis", 
                 "Pentose phosphate pathway", 
                 "Citrate cycle (TCA cycle)", 
                 "Pyruvate metabolism", 
                 "Fructose and mannose metabolism")
#展示前10个
#input.pathway <- rownames(countexp.Seurat_MP@assays$METABOLISM$score)[1:10]

dp <- DotPlot.metabolism(obj = countexp.Seurat_MP_VISION, 
                   pathway = input.pathway, 
                   phenotype = "group_celltype3",  #更改phenotype 参数
                   norm = "y")
ggsave(plot = dp, filename = "MP_metabolic_VISION.pdf", width = 10, height = 5)

cat("=== 巨噬细胞VISION代谢分析完成 ===\n")
cat("结果已保存到:", file.path(OUTPUT_DIR, "metabolic"), "\n")