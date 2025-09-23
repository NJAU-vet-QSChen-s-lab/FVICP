if(!require(limma))
if(!require(Seurat))
if(!require(dplyr))
if(!require(magrittr))
if(!require(devtools))
if(!require(ggplot2))
if(!require(scater))
if(!require(dplyr))
#BiocManager::install("org.Mm.eg.db")
#BiocManager::install("org.Hs.eg.db")
#BiocManager::install("org.Rn.eg.db")
if(!require(msigdbr))
if(!require(fgsea))
if(!require(clusterProfiler))
if(!require(biomaRt))
  library(limma)
library(Seurat)
library(dplyr)
library(magrittr)
library(devtools)
library(ggplot2)
library(scater)
#转录组可视化
#library(openxlsx)#读取.xlsx文件
library(ggplot2)#柱状图和点状图
library(stringr)#基因ID转换
library(enrichplot)#GO,KEGG,GSEA
library(clusterProfiler)#GO,KEGG,GSEA
library(GOplot)#弦图，弦表图，系统聚类图
library(DOSE)
library(ggnewscale)
library(topGO)#绘制通路网络图
library(circlize)#绘制富集分析圈图
library(ComplexHeatmap)#绘制图例
library(org.Rn.eg.db)
library(MetBrewer)
library(biomaRt)
#-----------------------------测试多核运算--------------------------------------
library("parallel")
makecore <- function(workcore,memory){
  if(!require(Seurat))install.packages('Seurat')
  if(!require(future))install.packages('future')
  plan("multisession", workers = workcore)
  options(future.globals.maxSize= memory*1024*1024**2)
}
cl <- detectCores()
makecore(cl, 16)
cl <- makeCluster(cl)
#insert your functions
plan("sequential")
stopCluster(cl)
rm(cl)