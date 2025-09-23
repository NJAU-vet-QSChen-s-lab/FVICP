# =============================================================================
# 巨噬细胞亚群追踪分析脚本
# =============================================================================
# 功能：巨噬细胞亚群提取、重新聚类、细胞类型注释，并进行CytoTRACE分化轨迹分析

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

# 读取Seurat对象
fascia <- qread(file.path(OUTPUT_DIR, "seurat", "fascia.qs"))
table(fascia$celltype2)

# 提取巨噬细胞亚群
MP <- subset(fascia, subset = celltype2 %in% c("MonC" ,"MP"))

MP <- seurat_process(MP, do_harmony = F)
MP$seurat_clusters <- MP$RNA_snn_res.0.1
MP@active.ident <- MP$seurat_clusters
qsave(MP, "MP.qs")

makecore <- function(workcore, memory){
  if(!require(Seurat)) install.packages('Seurat')
  if(!require(future)) install.packages('future')
  plan("multisession", workers = workcore)
  options(future.globals.maxSize= memory*1024*1024**2)
}

cl <- detectCores()
makecore(10, 64)
cl <- makeCluster(10)
dge <- FindAllMarkers(MP)
dge$diffpct <- dge$pct.1-dge$pct.2
write.csv(dge, "MP_markers_RNA_snn_res.0.1.csv")
plan("sequential")
stopCluster(cl)
rm(cl)

#进一步过滤一下不是巨噬细胞的部分，有少量成纤维
MP <- subset(MP, subset = seurat_clusters == "2", invert = T)
MP <- seurat_process(MP, do_harmony = F)
MP$seurat_clusters <- MP$RNA_snn_res.0.1
MP@active.ident <- MP$seurat_clusters

qsave(MP, "MP.qs")

sclist <- list(fascia, MP)
names(sclist) <- list("fascia", "MP")
names(sclist)

# 加载所需的包
library(foreach)
library(doParallel)

# 设置并行环境
numCores <- detectCores() # 获取可用的核心数
cl <- makeCluster(numCores) # 创建 cluster
registerDoParallel(cl) # 注册并行后端

# 并行计算: 使用 foreach 实现
pclist <- foreach(sce=sclist, .combine=list, .multicombine=TRUE, .packages = "findPC") %dopar% {
  find_PC(sce = sce, scale = F, nPC = c(30, 40, 50)) #这步时间很长，算完会返回一个肘子图
}
names(pclist) <- list("fascia", "MP")
saveRDS(pclist, "pclist.rds")
# 停止并行环境
stopCluster(cl)

MP <- seurat_process(MP, do_harmony = F)
MP$seurat_clusters <- MP$RNA_snn_res.0.1
MP@active.ident <- MP$seurat_clusters
MP <- RunPCA(MP, npcs = 50, verbose = FALSE)　
MP <- RunUMAP(MP, reduction =  "pca", dims = 1:13)
MP <- RunTSNE(MP, reduction = "pca", dims = 1:13) 
DimPlot(MP)
qsave(MP, "MP.qs")

makecore <- function(workcore, memory){
  if(!require(Seurat)) install.packages('Seurat')
  if(!require(future)) install.packages('future')
  plan("multisession", workers = workcore)
  options(future.globals.maxSize= memory*1024*1024**2)
}

cl <- detectCores()
makecore(10, 64)
cl <- makeCluster(10)
dge <- FindAllMarkers(MP)
write.csv(dge, "MP_markers_RNA_snn_res.0.1.csv")
plan("sequential")
stopCluster(cl)
rm(cl)

MP$celltype3 <- recode(MP$seurat_clusters, 
                       "0"="Gdf15+MP",
                       "1"="Cd163+MP",
                       "2"="Ccr2+MP",
                       "3"="Scimp+MP",
                       "4"="Ncald+DC",
                       "5"="Ly6c+MonC",
                       "6"="Clec9a+DC"
                       )
sclist <- SplitObject(MP, split.by = "sample")
namelist <- names(sclist)
for (i in 1:length(sclist)) {
  lt <- length(table(sclist[[i]]$celltype3))
  for (j in lt) {
    sclist[[i]]$group_celltype3 <- paste(namelist[[i]], "_", sclist[[i]]$celltype3, sep = "")
  }
  table(sclist[[i]]$group_celltype3)
}
table(sclist[[1]]$group_celltype3)
table(sclist[[2]]$group_celltype3)
test <- merge(sclist[[1]], y = sclist[[2]])
table(test$group_celltype3)
rm(sclist)
MP$group_celltype3 <- test$group_celltype3
rm(test)

sclist <- SplitObject(MP, split.by = "celltype3")
names(sclist)
#[1] "Cd163+MP"  "Ccr2+MP"   "Clec9a+DC" "Ncald+DC"  "Scimp+MP"  "Gdf15+MP"  "Ly6c+MonC"

sclist <- SplitObject(MP, split.by = "sample")
namelist <- names(sclist)
for (i in 1:length(sclist)) {
  lt <- length(table(sclist[[i]]$celltype2))
  for (j in lt) {
    sclist[[i]]$group_celltype2 <- paste(namelist[[i]], "_", sclist[[i]]$celltype2, sep = "")
  }
  table(sclist[[i]]$group_celltype2)
}
table(sclist[[1]]$group_celltype2)
table(sclist[[2]]$group_celltype2)
test <- merge(sclist[[1]], y = sclist[[2]])
table(test$group_celltype2)
rm(sclist)
MP$group_celltype2 <- test$group_celltype2
rm(test)

qsave(MP, "MP.qs")

# 更新主要的fascia对象
fascia <- qread(file.path(OUTPUT_DIR, "seurat", "fascia.qs"))
test <- subset(fascia, subset = celltype2 %in% c("MonC" ,"MP"), invert = T)

table(test$celltype2)
table(test$group_celltype2)
test <- merge(test, y = MP)
table(test$celltype2)
table(test$group_celltype2)

fascia$celltype2 <- test$celltype2
fascia$group_celltype2 <- test$group_celltype2
table(fascia$celltype2)
table(fascia$group_celltype2)
qsave(fascia, file.path(OUTPUT_DIR, "seurat", "fascia.qs"))

rm(test)

#跑一下分化确认一下
setwd(PROJECT_ROOT)
dir.create(file.path(OUTPUT_DIR, "metabolic"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUTPUT_DIR, "metabolic", "trace_MP"), showWarnings = FALSE, recursive = TRUE)
setwd(file.path(OUTPUT_DIR, "metabolic", "trace_MP"))

library(CytoTRACE)
MP_ct <- as.matrix(MP@assays$RNA@counts)
MP_results <- CytoTRACE(mat = MP_ct,  enableFast = T, subsamplesize = 8600,
                        ncores = 8)
plotCytoTRACE(MP_results, phenotype = MP$group_celltype2, 
              emb = MP@reductions$umap@cell.embeddings, outputDir = './')
plotCytoGenes(MP_results, numOfGenes = 10, outputDir = './')

cat("=== 巨噬细胞亚群追踪分析完成 ===\n")
cat("结果已保存到:", file.path(OUTPUT_DIR, "metabolic"), "\n")

