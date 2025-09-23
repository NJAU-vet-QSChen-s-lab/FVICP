# =============================================================================
# 差异表达分析脚本
# =============================================================================
# 功能：进行差异表达分析

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

# 创建DGE目录
dir.create(file.path(OUTPUT_DIR, "DGE"), showWarnings = FALSE, recursive = TRUE)
setwd(file.path(OUTPUT_DIR, "DGE"))

# 读取Seurat对象
fascia <- qread(file.path(OUTPUT_DIR, "seurat", "fascia.qs"))
table(fascia$orig.ident)
table(fascia$sample)
table(fascia$celltype)

sclist <- SplitObject(fascia, split.by = "sample")
namelist <- names(sclist)
for (i in 1:length(sclist)) {
  lt <- length(table(sclist[[i]]$celltype))
  for (j in lt) {
    sclist[[i]]$group_celltype <- paste(namelist[[i]], "_", sclist[[i]]$celltype, sep = "")
  }
  table(sclist[[i]]$group_celltype)
}
table(sclist[[1]]$group_celltype)
table(sclist[[2]]$group_celltype)
test <- merge(sclist[[1]], y = sclist[[2]])
table(test$group_celltype)
rm(sclist)

fascia$group_celltype <- test$group_celltype
rm(test)

table(fascia$celltype)
fascia@active.ident <- as.factor(fascia$sample)

fascia <- subset(fascia, subset = celltype == "MuSC", invert = T)

sclist <- SplitObject(fascia, split.by = "celltype")
namelist <- names(sclist)
namelist

# 加载所需的包
library(foreach)
library(doParallel)

# 设置并行环境
numCores <- detectCores() # 获取可用的核心数
cl <- makeCluster(numCores) # 创建 cluster
registerDoParallel(cl) # 注册并行后端

# 并行计算: 使用 foreach 实现
DGElist <- foreach(sce=sclist, .combine=list, .multicombine=TRUE, .packages = "Seurat") %dopar% {
  FindMarkers(sce, group.by = "sample", ident.1 = "dysentery", ident.2 = "health", logfc.threshold = 0,
              min.cells.feature = 0, min.cells.group = 0)
}

# 停止并行环境
stopCluster(cl)

for (i in seq_along(sclist)) {
  write.csv(DGElist[i], file = paste0(namelist[i], "_dge_dys_vs_health.csv"))
}

# 获取DGElist的名称列表
names(DGElist) <- names(sclist)
namelist <- names(DGElist)
namelist

# 遍历DGElist列表中的每一个元素
for (i in 1:length(DGElist)) {
  # 获取当前元素
  current_list <- DGElist[[i]]
  #GSEA
  gsedf <- current_list$avg_log2FC
  #gsedf <- gsedf[ , 1] #表达值或者p值 或者monocle3的markerscore
  names(gsedf) = as.character(rownames(current_list)) #SYMBOL                                                                                                                                                                                                                     
  gsedf <- sort(gsedf, decreasing = T) #一定做个排序
  gse_BP<- gseGO(gsedf, ont = "BP", OrgDb = 'org.Rn.eg.db', pvalueCutoff = 1, keyType = "SYMBOL")
  gse_bp <- data.frame(gse_BP)
  write.csv(gse_bp, paste0("GSEA_BP_", namelist[i], ".csv"))
}

cat("=== 差异表达分析完成 ===\n")
cat("结果已保存到:", file.path(OUTPUT_DIR, "DGE"), "\n")
  
  