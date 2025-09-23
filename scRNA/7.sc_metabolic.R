# =============================================================================
# 单细胞代谢通路分析脚本
# =============================================================================
# 功能：使用AUCell方法计算KEGG代谢通路活性

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

# 读取Seurat对象
fascia <- qread(file.path(OUTPUT_DIR, "seurat", "fascia.qs"))

# 创建代谢分析目录
dir.create(file.path(OUTPUT_DIR, "metabolic"), showWarnings = FALSE, recursive = TRUE)

# 代谢通路分析代码...
# [此处保留原有的代谢通路分析逻辑]

cat("=== 代谢通路分析完成 ===\n")

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

sce3 <- fascia
genes <- rownames(fascia@assays[["RNA"]])
genes <- toupper(genes)
genes
rownames(sce3@assays[["RNA"]]@counts) <- genes
rownames(sce3@assays[["RNA"]]@data) <- genes
countexp.Seurat <- sc.metabolism.Seurat(obj = sce3,  #Seuratde单细胞object
                                        method = "AUCell", 
                                        imputation = F, 
                                        ncores = 8, 
                                        metabolism.type = "KEGG")
score <- countexp.Seurat@assays$METABOLISM$score
score[1:4,1:4]

#将score中barcode的点转为下划线
score_change <- score %>% 
  select_all(~str_replace_all(., "\\.", "-"))  #基因ID不规范会报错,下划线替换-
#确定细胞barcode椅子
identical(colnames(score_change) , rownames(countexp.Seurat@meta.data))
#[1] TRUE
countexp.Seurat@meta.data <- cbind(countexp.Seurat@meta.data,t(score_change) )

#可以直接使用Seurat的相关函数
p1 <- FeaturePlot(countexp.Seurat,features = "Glycolysis / Gluconeogenesis")
p2 <- VlnPlot(countexp.Seurat,features = "Glycolysis / Gluconeogenesis", group.by = "celltype2")
p1
p2
p1 + p2

qsave(countexp.Seurat, "countexp.Seurat.qs")
countexp.Seurat <- qread("countexp.Seurat.qs")

#直接指定
input.pathway<-c("Glycolysis / Gluconeogenesis", 
                 "Oxidative phosphorylation", 
                 "Citrate cycle (TCA cycle)")
#展示前10个
#input.pathway <- rownames(countexp.Seurat@assays$METABOLISM$score)[1:10]

DotPlot.metabolism(obj = countexp.Seurat, 
                   pathway = input.pathway, 
                   phenotype = "group_celltype2",  #更改phenotype 参数
                   norm = "y")

